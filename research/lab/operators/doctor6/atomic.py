"""Expert-triple atomicity.

The runtime's execution unit is the expert TRIPLE (gate_proj, up_proj, down_proj).
Both ensure_device_expert_table and the control path refuse a partial triple closed.
Per-organ selection previously produced 841 partial triples and the artifact would
not admit at all.

Enforce atomicity at the LAST stage that can violate it (after per-organ compose;
the greedy pack can re-create partials after selection has cleared them).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

COMPONENTS: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")


def triple_key(layer: int, expert: int) -> tuple[int, int]:
    return (int(layer), int(expert))


def group_by_triple(
    organs: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, int], dict[str, Mapping[str, Any]]]:
    groups: dict[tuple[int, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in organs:
        k = triple_key(int(row["layer"]), int(row["expert"]))
        groups[k][str(row["component"])] = row
    return groups


def find_partial_triples(
    organs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    partials = []
    for (L, e), comps in group_by_triple(organs).items():
        missing = [c for c in COMPONENTS if c not in comps]
        if missing:
            partials.append(
                {
                    "layer": L,
                    "expert": e,
                    "present": sorted(comps.keys()),
                    "missing": missing,
                }
            )
    return partials


def enforce_atomic_chains(
    prescriptions: list[MutableMapping[str, Any]],
) -> dict[str, Any]:
    """Force each (layer, expert) triple to share a closed chain.

    Strategy: within a triple, take the UNION of rungs (max effort wins) so the
    weakest organ is not under-treated, and mark every member with the same
    atomic_chain. Partials (missing a component in the sample) are flagged —
    they must not be sealed as a device expert table entry.
    """
    groups: dict[tuple[int, int], list[MutableMapping[str, Any]]] = defaultdict(list)
    for row in prescriptions:
        groups[triple_key(int(row["layer"]), int(row["expert"]))].append(row)

    n_closed = 0
    n_partial = 0
    n_unified = 0
    partials: list[dict[str, Any]] = []

    for (L, e), rows in groups.items():
        present = {str(r["component"]) for r in rows}
        missing = [c for c in COMPONENTS if c not in present]
        if missing:
            n_partial += 1
            partials.append(
                {
                    "layer": L,
                    "expert": e,
                    "present": sorted(present),
                    "missing": missing,
                    "seal_admissible": False,
                }
            )
            for r in rows:
                r["atomic_triple"] = {
                    "closed": False,
                    "missing": missing,
                    "seal_admissible": False,
                }
            continue

        # Union of chains: preserve order from a canonical rung order.
        from lab.operators.doctor6.rungs import RUNG_ORDER

        order_index = {n: i for i, n in enumerate(RUNG_ORDER)}
        union: list[str] = []
        seen: set[str] = set()
        # Prefer the longest chain's order, then add missing.
        rows_sorted = sorted(rows, key=lambda r: -len(r.get("chain") or []))
        for r in rows_sorted:
            for step in r.get("chain") or []:
                if step not in seen:
                    seen.add(step)
                    union.append(step)
        union.sort(key=lambda n: order_index.get(n, 1000))

        changed = False
        for r in rows:
            per_organ = list(r.get("chain") or [])
            # Preserve per-organ adaptive chain; pack with the closed atomic union.
            r["per_organ_chain"] = per_organ
            r["atomic_chain"] = list(union)
            if per_organ != union:
                changed = True
            # `chain` remains the per-organ prescription (what was measured);
            # treat/pack uses atomic_chain when seal_admissible.
            r["atomic_triple"] = {
                "closed": True,
                "missing": [],
                "seal_admissible": True,
                "unified_from_per_organ": changed,
                "atomic_chain": list(union),
            }
        n_closed += 1
        if changed:
            n_unified += 1

    return {
        "n_triples": len(groups),
        "n_closed": n_closed,
        "n_partial": n_partial,
        "n_unified_to_union_chain": n_unified,
        "partials": partials,
        "seal_admissible": n_partial == 0,
        "rule": (
            "expert triple is the runtime execution unit; partial triples are "
            "never seal-admissible; closed triples share the union chain so "
            "down_proj escalation is not dropped by a weaker sibling"
        ),
    }
