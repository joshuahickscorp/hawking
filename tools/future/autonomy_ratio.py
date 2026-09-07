"""Supervisor interventions per accepted result, derived from disk, never self-reported.

G036 requires this ratio be tracked EXPLICITLY and trend DOWN across specimens.
There was no counter anywhere, and the tempting way to build one is to let the
supervisor state its own intervention count -- which measures candour, not
autonomy.

So every term comes from a durable artifact neither Claude nor HCLI can restate:

  SUPERVISOR interventions  commits on the campaign branch whose author is the
                            human/Claude pair and which touch tools/ or hcli/
  GROK interventions        task directories under ~/.claude-grok/tasks, each of
                            which is one dispatched lane
  HCLI work                 receipts HCLI itself wrote, identified by the
                            accepted-work markers in its own mission state
  ACCEPTED RESULTS          axes in the Odyssey ledger that moved to MEASURED or
                            REFUSED *and carry a receipt path*

A dispatched Grok lane still counts as a supervisor intervention. Delegating the
typing does not make the campaign autonomous -- the supervisor still chose the
target, and G036 is about who CHOOSES, not who types. Counting lanes as free
would let the ratio be improved by fanning out harder, which is exactly the
self-deception this instrument exists to prevent.
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Any

CAMPAIGN_PATHS = ("tools/", "hcli/")
GROK_TASKS = os.path.expanduser("~/.claude-grok/tasks")


class RatioRefused(RuntimeError):
    pass


def _git(repo: str, *args: str) -> str:
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, timeout=60)
    if r.returncode:
        raise RatioRefused(f"git {' '.join(args)} failed: {r.stderr.strip()[:200]}")
    return r.stdout


def supervisor_commits(repo: str, since: str) -> list[dict]:
    """Commits touching campaign code since a revision. One commit, one intervention."""
    out = _git(repo, "log", "--no-merges", "--format=%H%x1f%at%x1f%s", f"{since}..HEAD")
    rows = []
    for line in out.splitlines():
        if not line.strip():
            continue
        sha, ts, subj = line.split("\x1f", 2)
        files = _git(repo, "show", "--name-only", "--format=", sha).split()
        if any(f.startswith(CAMPAIGN_PATHS) for f in files):
            rows.append({"sha": sha[:9], "at": int(ts), "subject": subj,
                         "files": len(files)})
    return rows


def grok_lanes(since_epoch: int, root: str | None = None) -> list[dict]:
    """Each task directory is one dispatched lane."""
    GROK = root or GROK_TASKS
    if not os.path.isdir(GROK):
        return []
    out = []
    for name in os.listdir(GROK):
        d = os.path.join(GROK, name)
        meta = os.path.join(d, "metadata.json")
        if not os.path.isfile(meta):
            continue
        try:
            st = os.stat(meta).st_mtime
        except OSError:
            continue
        if st >= since_epoch:
            out.append({"task": name, "at": int(st)})
    return out


def accepted_results(ledger_path: str) -> dict:
    """An accepted result is a DISTINCT EVIDENCE ARTIFACT, not a resolved axis.

    Counting axes was the first version and it is gameable to the point of
    meaninglessness: one census run resolved 49 axes against a single receipt and
    one rule resolved 84 more against 14 reasons, so 150 "results" came from 15
    artifacts. Since the ratio is interventions over results, counting axes made
    a single tool invocation look like 150 accepted results and improved the
    score tenfold for no additional science.

    A MEASURED axis is attributable only if its receipt EXISTS INSIDE THE REPO. A
    path under /tmp or a session scratchpad is not durable evidence -- it will be
    gone before anyone checks it -- and the first ledger seeded here cited exactly
    such a path.
    """
    led = json.load(open(ledger_path))
    repo = os.path.dirname(os.path.abspath(ledger_path))
    while repo != "/" and not os.path.isdir(os.path.join(repo, ".git")):
        repo = os.path.dirname(repo)
    receipts: set = set()
    reasons: set = set()
    axes = unattributable = 0
    ephemeral: set = set()
    for rec in led["specimens"]:
        for axis, a in rec["axes"].items():
            if a["state"] == "MEASURED":
                axes += 1
                r = a.get("receipt")
                if not r:
                    unattributable += 1
                elif not os.path.isfile(os.path.join(repo, r) if not os.path.isabs(r) else r):
                    ephemeral.add(r)
                    unattributable += 1
                elif os.path.isabs(r) and not r.startswith(repo + os.sep):
                    ephemeral.add(r)
                    unattributable += 1
                else:
                    receipts.add(os.path.relpath(r, repo) if os.path.isabs(r) else r)
            elif a["state"] == "REFUSED":
                axes += 1
                if a.get("reason"):
                    reasons.add(a["reason"][:120])
                else:
                    unattributable += 1
    return {"axes_resolved": axes,
            "attributable": len(receipts) + len(reasons),
            "distinct_receipts": len(receipts),
            "distinct_reasons": len(reasons),
            "unattributable_axes": unattributable,
            "ephemeral_receipts": sorted(ephemeral)}


def ratio(repo: str, ledger_path: str, since: str, since_epoch: int,
          grok_root: str | None = None) -> dict:
    sup = supervisor_commits(repo, since)
    lanes = grok_lanes(since_epoch, grok_root)
    acc = accepted_results(ledger_path)
    interventions = len(sup) + len(lanes)
    if acc["attributable"] == 0:
        raise RatioRefused(
            "zero attributable results: the ratio would be a division by zero dressed "
            "up as infinite supervision. Record a result with a receipt first.")
    # G026 asks for progress per WALL per RESOURCE per intervention. Wall comes
    # from the commit timestamps that bound the window -- it is the only term of
    # the four that is already recorded by something other than this module.
    # A degenerate wall must REFUSE, not divide by 1e-9 and report a billion
    # artifacts an hour. That is exactly what the epsilon here did: with no
    # commits in the window it printed artifacts_per_wall_hour 1000000000.0,
    # a plausible-shaped number standing in for an unmeasured quantity.
    stamps = sorted(c["at"] for c in sup)
    wall_s = (stamps[-1] - since_epoch) if stamps else 0
    wall_h = wall_s / 3600.0 if wall_s >= 60 else None

    return {
        "supervisor_commits": len(sup),
        "grok_lanes_dispatched": len(lanes),
        "interventions": interventions,
        "accepted_results": acc["attributable"],
        "axes_resolved": acc["axes_resolved"],
        "distinct_receipts": acc["distinct_receipts"],
        "distinct_reasons": acc["distinct_reasons"],
        "unattributable_axes": acc["unattributable_axes"],
        "ephemeral_receipts": acc["ephemeral_receipts"],
        "interventions_per_accepted_result": round(interventions / acc["attributable"], 4),
        "wall_hours": round(wall_h, 3) if wall_h else None,
        "artifacts_per_wall_hour": (round(acc["attributable"] / wall_h, 2)
                                    if wall_h else None),
        "artifacts_per_wall_hour_per_intervention": (
            round(acc["attributable"] / wall_h / max(interventions, 1), 4)
            if wall_h else None),
        "wall_gap": (None if wall_h else
                     f"window spans {wall_s}s of commits, under the 60s floor: a rate "
                     f"over a wall that short is division noise, not throughput"),
        "resource_term": None,
        "resource_gap": (
            "G026's metric is progress per wall per RESOURCE per intervention, and the "
            "resource term is NOT MEASURED. It is left null rather than substituted, "
            "because a throughput figure computed over three of four terms and reported "
            "as the metric is exactly the arithmetic G026 forbids. THE EXACT MISSING "
            "MECHANISM: campaign_memory_guard.sample() already reads free pages, "
            "compressor size, swapfile count and RSS, but nothing calls it at the START "
            "and END of an experiment and writes the delta into the receipt. Until a "
            "measured run brackets itself with two samples, resource is unknown and "
            "saying so is the honest reading."),
        "note": ("a dispatched lane counts as a supervisor intervention: delegating the "
                 "typing does not make the campaign autonomous, and counting lanes as "
                 "free would let the ratio improve by fanning out harder"),
    }


def _selfcheck() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        led = {"schema": "odyssey-ledger-1", "specimens": [{
            "slug": "a--b", "axes": {
                "anatomy": {"state": "MEASURED", "receipt": "r.json", "reason": None},
                "ebpw": {"state": "MEASURED", "receipt": None, "reason": None},
                "gpu": {"state": "REFUSED", "receipt": None, "reason": "no mlx module for this"},
                "cpu": {"state": "REFUSED", "receipt": None, "reason": None},
                "tps": {"state": "OWED", "receipt": None, "reason": None},
            }}]}
        p = os.path.join(d, "l.json")
        json.dump(led, open(p, "w"))
        a = accepted_results(p)
        # 4 axes resolved. The MEASURED one cites r.json which does not exist, so
        # it is EPHEMERAL, not evidence. One REFUSED carries a reason, one does
        # not. Attributable = 1 (a single distinct reason), never 4.
        assert a["axes_resolved"] == 4, a
        assert a["attributable"] == 1, a
        assert a["unattributable_axes"] == 3, a
        assert a["ephemeral_receipts"] == ["r.json"], a

        # Two axes sharing ONE receipt are ONE accepted result, not two.
        open(os.path.join(d, "real.json"), "w").write("{}")
        os.makedirs(os.path.join(d, ".git"), exist_ok=True)
        led2 = {"schema": "odyssey-ledger-1", "specimens": [{"slug": "a", "axes": {
            "anatomy": {"state": "MEASURED", "receipt": "real.json", "reason": None},
            "ebpw": {"state": "MEASURED", "receipt": "real.json", "reason": None},
            "gpu": {"state": "MEASURED", "receipt": "real.json", "reason": None}}}]}
        p2 = os.path.join(d, "l2.json")
        json.dump(led2, open(p2, "w"))
        a2 = accepted_results(p2)
        assert a2["axes_resolved"] == 3 and a2["attributable"] == 1, a2

        # An ABSOLUTE path outside the repo is ephemeral even when the file
        # exists -- the first ledger cited a session scratchpad that will vanish.
        outside = os.path.join(tempfile.gettempdir(), "outside_the_repo.json")
        open(outside, "w").write("{}")
        led3 = {"schema": "odyssey-ledger-1", "specimens": [{"slug": "a", "axes": {
            "anatomy": {"state": "MEASURED", "receipt": outside, "reason": None}}}]}
        p3 = os.path.join(d, "l3.json")
        json.dump(led3, open(p3, "w"))
        a3 = accepted_results(p3)
        assert a3["attributable"] == 0, a3
        assert a3["ephemeral_receipts"] == [outside], a3
        os.unlink(outside)

        # A dispatched lane MUST count as a supervisor intervention. Counting
        # lanes as free halves the ratio on real data (0.1765 against 0.3529).
        gk = os.path.join(d, "grok_tasks")
        for i in range(3):
            os.makedirs(os.path.join(gk, f"task{i}"), exist_ok=True)
            open(os.path.join(gk, f"task{i}", "metadata.json"), "w").write("{}")
        assert len(grok_lanes(0, gk)) == 3, grok_lanes(0, gk)
        led4 = {"schema": "odyssey-ledger-1", "specimens": [{"slug": "a", "axes": {
            "gpu": {"state": "REFUSED", "receipt": None,
                    "reason": "no mlx_lm module exists for this architecture"}}}]}
        p4 = os.path.join(d, "l4.json")
        json.dump(led4, open(p4, "w"))
        rr = ratio(os.getcwd(), p4, "HEAD", 0, grok_root=gk)
        assert rr["grok_lanes_dispatched"] == 3, rr
        assert rr["interventions"] >= 3, rr

        # G026's fourth term must be null-or-measured, never both and never
        # quietly filled. A throughput number computed over three of four terms
        # and presented as the metric is the arithmetic G026 forbids.
        assert (rr["resource_term"] is None) == bool(rr["resource_gap"]), rr
        # A degenerate wall reports nothing rather than a billion per hour.
        assert rr["wall_hours"] is None and rr["artifacts_per_wall_hour"] is None, rr
        assert "division noise" in rr["wall_gap"], rr
        if rr["resource_term"] is None:
            assert "EXACT MISSING MECHANISM" in rr["resource_gap"], rr
            assert "artifacts_per_wall_hour_per_resource" not in rr, (
                "a per-resource figure was reported while resource is unmeasured")

        # Zero attributable results must REFUSE, not report infinite autonomy.
        json.dump({"schema": "odyssey-ledger-1", "specimens": [{"slug": "x", "axes": {
            "anatomy": {"state": "OWED", "receipt": None, "reason": None}}}]}, open(p, "w"))
        try:
            ratio(".", p, "HEAD", 0)
            raise AssertionError("reported a ratio with no attributable results")
        except RatioRefused as e:
            assert "division by zero" in str(e)
    print("selfcheck OK")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        print(json.dumps(ratio(".", sys.argv[1], sys.argv[2], int(sys.argv[3])), indent=1))
