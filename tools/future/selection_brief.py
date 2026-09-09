"""The five inputs G018 says HCLI must be GIVEN before it selects. (G018)

G018's acceptance: "HCLI is given Pareto state, measurement debt, architecture
novelty, capability and expected information gain, and chooses -- with its
reason." Only ONE of those five is reachable from HCLI's tool surface today.
odyssey.ledger carries measurement debt. There is no registered pareto tool, no
capability tool, and nothing that reports architecture novelty --
campaign_pareto.py computes a 210-candidate, 70-point frontier and is imported
by two sidecar modules and no tool. Built, and out of reach of the thing that
needs it.

THIS BRIEF DOES NOT RANK, AND THAT IS DELIBERATE. G018 exists because Claude
must stop naming which organism goes next; a "brief" that sorted by desirability
would name it in a different font. Rows come back in slug order, which carries
no opinion, and every field is a measured fact or an explicit absence. The
choice, and the reason, are HCLI's.

Architecture novelty is reported as `family_siblings` -- how many other
specimens in the lake share this body's family. That is a count, not a score:
whether "few siblings" means valuable novelty or an unsupported dead end is
exactly the judgement being reserved.

Expected information gain is NOT computed. It is a judgement over the other
four, and computing it here would be the ranking this file refuses to do.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO / "receipts" / "future" / "G034_ODYSSEY_LEDGER.json"
CATALOG = REPO / "receipts" / "future" / "modellake-index" / "catalog.json"


def _census_rows() -> Dict[str, Dict[str, Any]]:
    """klass, complete EBPW and size from lake.census, the authoritative producer.

    The raw catalog carries `architecture_family` and `bytes` and knows nothing
    about klass or complete EBPW -- those are computed by the census. Reading the
    catalog for them returned None on every row and "unknown" for every family,
    so the first version of this brief supplied two of its five inputs as blanks.
    """
    sys.path.insert(0, str(REPO))
    try:
        from hcli.tool_registry import default_tool_registry
        reg = default_tool_registry(str(REPO), repo_root=str(REPO))
        res = reg.invoke("lake.census", {"limit": 200})
        val = res.value if hasattr(res, "value") else res
        return {r["slug"]: r for r in (val.get("rows") or [])}
    except Exception:
        return {}


def _pareto_organisms() -> Dict[str, Any]:
    """Which organisms campaign_pareto has scored, and how many points survive."""
    sys.path.insert(0, str(REPO / "tools" / "future"))
    try:
        import campaign_pareto as cp
        h = cp.harvest()
    except Exception as exc:
        return {"available": False, "why": f"{type(exc).__name__}: {exc}"[:160]}
    # "is this organism on the frontier" is TRUE for nearly every organism --
    # 70 frontier POINTS spread across ~59 organisms -- so as a boolean it
    # separates nothing. What varies, and therefore what informs, is how many of
    # an organism's points survive domination out of how many it has.
    on = Counter(p.get("organism") for p in (h.get("frontier") or [])
                 if isinstance(p, dict) and p.get("organism"))
    total = {k: (len(v) if hasattr(v, "__len__") else v)
             for k, v in (h.get("by_organism") or {}).items()}
    return {
        "available": True,
        "n_candidates": h.get("n_candidates"),
        "n_frontier": h.get("n_frontier"),
        "frontier_points_by_organism": dict(on),
        "points_by_organism": total,
    }


def brief(limit: int = 12) -> Dict[str, Any]:
    led = json.loads(LEDGER.read_text())
    cat = {}
    if CATALOG.exists():
        try:
            cat = {r["slug"]: r for r in json.loads(CATALOG.read_text())["specimens"]}
        except Exception:
            cat = {}
    cen = _census_rows()
    par = _pareto_organisms()
    families = Counter(
        (cat.get(r["slug"], {}).get("architecture_family") or "unknown")
        for r in led["specimens"]
    )

    rows: List[Dict[str, Any]] = []
    for rec in led["specimens"]:
        slug = rec["slug"]
        c = cat.get(slug, {})
        x = cen.get(slug, {})
        owed = sorted(a for a, v in rec["axes"].items() if v["state"] == "OWED")
        refused = sorted(a for a, v in rec["axes"].items() if v["state"] == "REFUSED")
        fam = c.get("architecture_family") or "unknown"
        cap = rec["axes"].get("nr_candidate", {}).get("state")
        rows.append({
            "slug": slug,
            # measurement debt
            "owed_axes": owed,
            "n_owed": len(owed),
            "refused_axes": refused,
            # cost
            "gib": x.get("gib") or (round(c["bytes"] / 2**30, 1) if c.get("bytes") else None),
            "complete_ebpw": x.get("complete_ebpw"),
            "blocked_by": x.get("blocked_by") or None,
            # architecture novelty, as a COUNT not a score
            "family": fam,
            "klass": x.get("klass"),
            "family_siblings": families.get(fam, 0) - 1,
            # capability: does this body have a verdict at all
            "capability_axis_state": cap,
            # pareto
            "pareto_points_on_frontier": (par.get("frontier_points_by_organism", {}).get(slug, 0)
                                          if par.get("available") else None),
            "pareto_points_total": (par.get("points_by_organism", {}).get(slug)
                                    if par.get("available") else None),
        })

    # HOW MUCH EACH INPUT ACTUALLY SEPARATES THESE SPECIMENS. A field that
    # takes one value across all 56 cannot inform a choice between them, and
    # presenting it beside fields that do would let it borrow their authority.
    # pareto_points_on_frontier is the live example: every scored organism has
    # exactly ONE point and every one of those points is non-dominated, because
    # with 11 axes and mostly-missing values almost nothing dominates anything.
    # It separates SCORED from UNSCORED and nothing else, and read as "on the
    # frontier" it would look like a quality signal it is not.
    discrimination = {}
    for field in ("n_owed", "gib", "klass", "family", "family_siblings",
                  "capability_axis_state", "pareto_points_on_frontier", "complete_ebpw"):
        vals = Counter(json.dumps(r.get(field), sort_keys=True) for r in rows)
        discrimination[field] = {
            "distinct_values": len(vals),
            "most_common": [json.loads(k) for k, _ in vals.most_common(3)],
            "informative": len(vals) > 2,
        }

    # NEUTRAL ORDER. Sorting by owed count, size or novelty would rank, and
    # ranking is the thing G018 takes away from Claude.
    rows.sort(key=lambda r: r["slug"])
    shown = rows[:max(1, limit)]
    return {
        # SCOPE FIRST. Engine._compact_closed_observations cuts an observation to
        # 500 chars as head 250 + tail 208, eliding the middle. This brief
        # serializes to ~14,600 chars, so anything after the first ~250 is
        # invisible to the round that called it. n / shown / truncated lead so a
        # caller still learns how many specimens exist and whether it has them
        # all -- the single thing it cannot reconstruct from the rows it can see.
        "n": len(rows),
        "shown": len(shown),
        "truncated": len(rows) > len(shown),
        "schema": "hawking.future.selection_brief.v1",
        "purpose": ("the five inputs G018 requires, assembled and NOT ranked. The choice and its "
                    "reason are yours."),
        "ordering": "slug, alphabetically -- deliberately carries no opinion",
        "does_not_include": ("expected information gain. It is a judgement over the other four, "
                             "and computing it here would be the ranking this tool refuses."),
        "n": len(rows),
        "shown": len(shown),
        "truncated": len(rows) > len(shown),
        "field_discrimination": discrimination,
        "read_this_first": (
            "pareto_points_on_frontier takes only two values across these specimens -- scored (1) "
            "or unscored (0). Every scored organism contributes exactly one point and every one "
            "of those points is non-dominated, so it marks WHETHER a body has been scored, NOT "
            "whether it scored well. klass is absent for 24 bodies the census has not classified."),
        "pareto_state": {k: v for k, v in par.items()
                         if k not in ("points_by_organism", "frontier_points_by_organism")},
        "rows": shown,
    }


def _selftest() -> List[str]:
    fails = []
    b = brief(limit=5)
    if b["n"] < 50:
        fails.append(f"only {b['n']} specimens -- the ledger should carry ~56")
    if len(b["rows"]) != 5:
        fails.append("limit not honoured")
    if not b["truncated"]:
        fails.append("truncation not disclosed while showing 5 of many")
    # The brief must not smuggle a ranking in through the ordering.
    slugs = [r["slug"] for r in brief(limit=999)["rows"]]
    if slugs != sorted(slugs):
        fails.append("rows are not in neutral slug order -- an ordering is a recommendation")
    # Every one of the four computable inputs must be PRESENT on some row, or
    # the brief is not actually supplying what G018 asks for.
    all_rows = brief(limit=999)["rows"]
    for field in ("owed_axes", "family_siblings", "capability_axis_state",
                  "pareto_points_on_frontier", "gib", "klass"):
        vals = [json.dumps(r.get(field), sort_keys=True) for r in all_rows]
        if all(r.get(field) in (None, [], "unknown") for r in all_rows):
            fails.append(f"{field} is empty on every row -- that input is not really supplied")
        # A field IDENTICAL on every row separates nothing. The first version of
        # this brief reported family "unknown" and gib None for all 56 and
        # passed a weaker check that only looked for emptiness.
        elif len(set(vals)) == 1:
            fails.append(f"{field} is IDENTICAL on all {len(all_rows)} rows -- it cannot "
                         f"inform a choice between them")
    return fails


if __name__ == "__main__":
    fails = _selftest()
    for f in fails:
        print("SELFTEST FAIL:", f)
    print(f"selftest: {'FAILED' if fails else 'PASS'}")
    if fails:
        raise SystemExit(1)
    b = brief(limit=6)
    print(f"\n{b['n']} specimens, showing {b['shown']}, truncated={b['truncated']}")
    print(f"pareto: {b['pareto_state']}")
    for r in b["rows"]:
        print(f"  {r['slug'][:46]:46s} owed={r['n_owed']} gib={r['gib']} "
              f"fam={r['family']}(+{r['family_siblings']}) "
              f"pareto={r['pareto_points_on_frontier']}/{r['pareto_points_total']}")
