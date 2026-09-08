"""Fold the dense-anatomy sweep into the Odyssey ledger.

G002 is not satisfied by a sweep receipt existing. It is satisfied when every
reachable DENSE body's anatomy axis in the Odyssey ledger is MEASURED with a
receipt, or REFUSED with a mechanism. This is the step that makes the sweep
count, and it refuses to record anything it cannot attribute.
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from odyssey_ledger import measured, refused, progress   # noqa: E402

LEDGER = "receipts/future/G034_ODYSSEY_LEDGER.json"
SWEEP = "receipts/future/G002_DENSE_ANATOMY_SWEEP.json"


def ingest(ledger_path: str = LEDGER, sweep_path: str = SWEEP,
           dry_run: bool = False) -> dict:
    led = json.load(open(ledger_path))
    sweep = json.load(open(sweep_path))
    by = {r["slug"]: r for r in led["specimens"]}
    n_meas = n_ref = n_skip = 0
    unknown: list = []
    for row in sweep.get("rows") or []:
        rec = by.get(row["slug"])
        if rec is None:
            unknown.append(row["slug"])
            continue
        if rec["axes"]["anatomy"]["state"] != "OWED":
            n_skip += 1
            continue
        status = row.get("status")
        if status == "ANATOMY":
            top = row.get("organ_ordering") or row.get("top_organ")
            measured(rec, "anatomy",
                     {"kind": "dense", "top_organ": top,
                      "wall_s": row.get("wall_s"),
                      "peak_rss_gib": row.get("peak_rss_gib"),
                      "cross_layer": row.get("cross_layer")},
                     sweep_path)
            n_meas += 1
        elif status in ("REFUSED", "GUARD_STOP"):
            why = str(row.get("refusal") or row.get("reason") or "")
            if len(why) < 20:
                raise ValueError(
                    f"{row['slug']}: {status} with no mechanism ({why!r}). A refusal "
                    f"without a named mechanism is an unmeasured axis wearing a "
                    f"finding's clothes -- refusing to record it.")
            refused(rec, "anatomy", f"dense sweep {status}: {why}")
            n_ref += 1
        else:
            raise ValueError(f"{row['slug']}: unknown sweep status {status!r}")
    if unknown:
        raise ValueError(f"sweep rows not in the ledger: {unknown[:5]}")
    if not dry_run:
        json.dump(led, open(ledger_path, "w"), indent=1)
    p = progress(led)
    return {"measured": n_meas, "refused": n_ref, "already_resolved": n_skip,
            "progress": p}


def _selfcheck() -> None:
    """A refusal without a mechanism must be REFUSED BY THE INGEST, not recorded."""
    import tempfile, os
    axes = ("anatomy", "ebpw", "gpu", "cpu", "tps", "nr_candidate", "nx_disposition")
    led = {"schema": "odyssey-ledger-1", "specimens": [
        {"slug": "a--b", "gib": 1.0, "family": "f", "class": "DENSE",
         "axes": {x: {"state": "OWED", "value": None, "reason": None, "receipt": None}
                  for x in axes}}]}
    with tempfile.TemporaryDirectory() as d:
        lp, sp = os.path.join(d, "l.json"), os.path.join(d, "s.json")
        json.dump(led, open(lp, "w"))
        json.dump({"rows": [{"slug": "a--b", "status": "REFUSED", "refusal": "n/a"}]},
                  open(sp, "w"))
        try:
            ingest(lp, sp)
            raise AssertionError("recorded a refusal with no mechanism")
        except ValueError as e:
            assert "no mechanism" in str(e), e

        json.dump({"rows": [{"slug": "a--b", "status": "ANATOMY", "wall_s": 1.0,
                             "organ_ordering": [{"organ": "q_proj"}]}]}, open(sp, "w"))
        out = ingest(lp, sp)
        assert out["measured"] == 1, out
        again = ingest(lp, sp)
        assert again["measured"] == 0 and again["already_resolved"] == 1, (
            "ingest is not idempotent -- a second run re-recorded a resolved axis")

        json.dump({"rows": [{"slug": "ghost--x", "status": "ANATOMY"}]}, open(sp, "w"))
        try:
            ingest(lp, sp)
            raise AssertionError("accepted a sweep row absent from the ledger")
        except ValueError as e:
            assert "not in the ledger" in str(e)
    print("selfcheck OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        print(json.dumps(ingest(dry_run="--dry-run" in sys.argv), indent=1))
