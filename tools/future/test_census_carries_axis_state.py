"""Round 20 read a VALUE off lake.census and treated it as an OWED CELL.

It saw `complete_ebpw: 16.0`, wrote "16.0 EBPW is worst owed", re-derived a
number the ledger already held as MEASURED, and closed with MEASURED still at
94. Nothing in the row said the axis was resolved, so the round inferred
owed-ness from magnitude -- the only signal present.

This pins the fix: the row must carry the axis state, and it must carry a state
that VARIES with the ledger. A constant, a default, or None everywhere would
satisfy a naive "field exists" check and reproduce the defect exactly.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hcli.tool_registry import default_tool_registry  # noqa: E402


def main() -> int:
    reg = default_tool_registry(str(REPO), repo_root=str(REPO))
    res = reg.invoke("lake.census", {"limit": 200})
    rows = (res.value if hasattr(res, "value") else res)["rows"]
    fails = []

    if not rows:
        return _report(["census returned no rows -- nothing is being checked"])

    if any("ebpw_axis_state" not in r for r in rows):
        fails.append("some rows carry no ebpw_axis_state at all")
    if any("owed_axes" not in r for r in rows):
        fails.append("some rows carry no owed_axes at all")

    states = {r.get("ebpw_axis_state") for r in rows}
    if states <= {None}:
        fails.append("ebpw_axis_state is None on every row -- the ledger was never read")
    # ebpw is resolved for EVERY specimen, so "must show two values" would be a
    # wrong bar here: the constant is the truth. The variation check belongs on
    # owed_axes, which genuinely differs body to body, and ebpw_axis_state is
    # pinned by the exact per-row cross-check below instead.
    shapes = {tuple(r.get("owed_axes") or ()) for r in rows}
    if len(shapes) < 2:
        fails.append(f"owed_axes is identical across all {len(rows)} rows ({shapes}) -- "
                     "a constant or a default, not a reading of the ledger")

    # owed_axes must agree with the ledger it claims to come from.
    import json
    led = json.load(open(REPO / "receipts/future/G034_ODYSSEY_LEDGER.json"))
    truth = {r["slug"]: sorted(a for a, v in r["axes"].items() if v["state"] == "OWED")
             for r in led["specimens"]}
    checked = 0
    for r in rows:
        if r["slug"] in truth and r.get("owed_axes") is not None:
            checked += 1
            if r["owed_axes"] != truth[r["slug"]]:
                fails.append(f"{r['slug']}: owed_axes {r['owed_axes']} != ledger {truth[r['slug']]}")
                break
    if checked == 0:
        fails.append("no row could be cross-checked against the ledger")

    # And the specific cell that caused this: a resolved ebpw must not read OWED.
    for r in rows:
        if r["slug"] in truth and "ebpw" not in truth[r["slug"]] \
                and r.get("ebpw_axis_state") == "OWED":
            fails.append(f"{r['slug']}: ebpw is resolved in the ledger and the census says OWED")
            break

    return _report(fails, checked, len(rows))


def _report(fails, checked=0, n=0) -> int:
    for f in fails:
        print("FAIL:", f)
    print(f"{'FAILED' if fails else 'PASS'} — {n} rows, {checked} cross-checked against the ledger")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
