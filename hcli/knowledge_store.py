"""Cross-specimen compounding memory (S008 sections 10-12).

The point is not to remember things. It is to make specimen N+1 cheaper than
specimen N WITHOUT turning one model's lesson into a universal law. Both failure
modes are real: forgetting costs a rediscovery, overgeneralising costs a wrong
prior that a whole campaign then inherits.

So scope is enforced structurally rather than by convention:

  SPECIMEN          true of this body. Carries nothing forward by itself.
  FAMILY_PRIOR      a HYPOTHESIS for the same architecture family. Must be
                    re-tested on arrival; it changes the search ORDER, never
                    the verdict.
  TRANSFERABLE_LAW  measured on >= 2 distinct specimens. Refused below that.
  MACHINE_LAW       a property of the host/runtime, not of any model.

Every entry needs a measurement and a reopen condition. A finding that cannot
say what would overturn it is an opinion.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

SPECIMEN = "SPECIMEN"
FAMILY_PRIOR = "FAMILY_PRIOR"
TRANSFERABLE_LAW = "TRANSFERABLE_LAW"
MACHINE_LAW = "MACHINE_LAW"
SCOPES = (SPECIMEN, FAMILY_PRIOR, TRANSFERABLE_LAW, MACHINE_LAW)

STORE = Path(__file__).resolve().parents[1] / "receipts" / "future" / "ODYSSEY_KNOWLEDGE.json"
MIN_SPECIMENS_FOR_LAW = 2


class Overgeneralisation(ValueError):
    """Raised when a claim is filed at a scope its evidence does not support."""


def validate(entry: Mapping[str, Any]) -> None:
    for k in ("id", "scope", "claim", "measurement", "reopen_condition", "sources"):
        if not entry.get(k):
            raise ValueError(f"{entry.get('id', '?')}: missing {k}")
    scope = entry["scope"]
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope}")
    n = len({s["specimen"] for s in entry["sources"] if s.get("specimen")})
    if scope == TRANSFERABLE_LAW and n < MIN_SPECIMENS_FOR_LAW:
        raise Overgeneralisation(
            f"{entry['id']}: TRANSFERABLE_LAW from {n} specimen(s); "
            f"needs >= {MIN_SPECIMENS_FOR_LAW}. File it as FAMILY_PRIOR."
        )
    if scope == FAMILY_PRIOR and not entry.get("architecture_condition"):
        raise ValueError(f"{entry['id']}: FAMILY_PRIOR needs an architecture_condition")
    if scope == MACHINE_LAW and not entry.get("host"):
        raise ValueError(f"{entry['id']}: MACHINE_LAW needs a host")
    if not entry.get("oracle_strength"):
        raise ValueError(f"{entry['id']}: needs oracle_strength")


def load() -> list[dict[str, Any]]:
    if not STORE.exists():
        return []
    return json.loads(STORE.read_text())["entries"]


def save(entries: Iterable[Mapping[str, Any]]) -> None:
    es = list(entries)
    for e in es:
        validate(e)
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(
        {"schema": "hawking.odyssey.knowledge.v1", "n_entries": len(es), "entries": es},
        indent=1) + "\n")


def _family_ok(entry: Mapping[str, Any], family: str) -> bool:
    """An entry with no architecture_condition applies to every family. One WITH
    a condition applies only inside it -- a machine law about top-k MoE decode
    must not be handed to a dense specimen just because the host matches."""
    cond = entry.get("architecture_condition")
    if not cond:
        return True
    return family in cond.get("families", [])


def priors_for(architecture_family: str, host: str | None = None) -> list[dict[str, Any]]:
    """What a NEW specimen starts from. Never SPECIMEN-scoped findings: those
    belong to the body that produced them and carry nothing forward."""
    out = []
    for e in load():
        if not _family_ok(e, architecture_family):
            continue
        if e["scope"] == TRANSFERABLE_LAW:
            out.append(e)
        elif e["scope"] == MACHINE_LAW:
            # host "any" is host-independent; otherwise it must match.
            if host is None or e.get("host") in (host, "any"):
                out.append(e)
        elif e["scope"] == FAMILY_PRIOR:
            out.append(e)     # _family_ok already enforced the condition
    return sorted(out, key=lambda e: (e["scope"] != MACHINE_LAW, e["id"]))
