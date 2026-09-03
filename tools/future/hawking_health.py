"""One command for the state of Hawking's research loop.

Each instrument below already exists and each was built because a number nobody
was looking at turned out to be wrong. Left as separate modules they are five
things to remember to run, which is the same as not having them.

    roadmap freshness      is the generated authority current with HEAD?
    capability census      what is BUILT, and on what evidence?
    acceptance integrity   do accepted gates still match their cited criterion?
    measurement provenance how many hardware numbers record their conditions?
    tool friction          what fraction of resident turns is lost to failed calls?
    odyssey wall           what can be said about campaign wall, and what cannot?

It measures nothing itself and changes nothing. Every number is read from a
committed artifact or recomputed from one, so running it is safe while a
protected measurement holds the GPU lane.

    python3 -m tools.future.hawking_health
    python3 -m tools.future.hawking_health --json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
GRAPH = REPO / "civilization" / "CAPABILITY_GRAPH.json"
STATE = REPO / "civilization" / "ROADMAP_STATE.json"


def _safe(fn, *a, **kw) -> Any:
    """An instrument that fails must say so, not take the dashboard down with it."""
    try:
        return fn(*a, **kw)
    except Exception as exc:                      # noqa: BLE001 - reported, not swallowed
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


def roadmap() -> dict[str, Any]:
    from tools.roadmap import regenerate
    graph = json.loads(GRAPH.read_text())
    state = json.loads(STATE.read_text())
    return {
        "fresh": _safe(regenerate.check) == 0,
        "generated_from_commit": graph.get("generated_from_commit"),
        "gates_by_status": graph["counts"]["gates_by_status"],
        "built": graph["counts"]["built_gates"],
        "accepted": graph["counts"]["accepted_gates"],
        "unresolved": state["TOTAL_UNRESOLVED_GATES"],
        "blocker_census": state["blocker_census"],
    }


def acceptance() -> dict[str, Any]:
    from tools.roadmap.auditor import criterion_matches_its_source
    counts: dict[str, int] = {}
    unverifiable = []
    for path in sorted((REPO / "receipts" / "acceptance").glob("*.json")):
        if "." in path.stem:
            continue
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(doc, dict) or str(doc.get("verdict", "")).upper() != "ACCEPTED":
            continue
        verdict, _why = criterion_matches_its_source(doc)
        counts[verdict] = counts.get(verdict, 0) + 1
        if verdict == "UNVERIFIABLE":
            unverifiable.append(path.stem)
    return {"criterion_recheck": counts, "unverifiable_gates": sorted(unverifiable)}


def provenance() -> dict[str, Any]:
    from tools.future import measurement_provenance_audit as A
    doc = A.audit()
    return {k: doc[k] for k in
            ("receipts_scanned", "receipts_carrying_a_hardware_number",
             "with_provenance", "without_provenance")}


def friction() -> dict[str, Any]:
    from tools.future import tool_friction as F
    doc = F.census()
    out = {k: doc[k] for k in
           ("tool_calls", "failed", "failure_rate", "model_wall_s",
            "tool_execution_share_of_wall")}
    out["file_not_found"] = doc["file_not_found"]
    out["cost"] = F.wasted_model_wall_s(doc)
    return out


def wall() -> dict[str, Any]:
    from tools.future import odyssey_wall as W
    serial = [c for c in W.COMPONENTS if c.parallel_by_construction is False]
    recordable = [c for c in serial if c.substrate_event is not None]
    return {
        "components": len(W.COMPONENTS),
        "on_serial_critical_path": len(serial),
        "of_those_recordable": len(recordable),
        "serial_coverage": round(len(recordable) / len(serial), 3) if serial else None,
        "measured_components": len(W.measured_components()),
        "projection": "REFUSED: no component has a measured duration",
    }


def report() -> dict[str, Any]:
    return {
        "schema": "hawking.future.health.v1",
        "roadmap": _safe(roadmap),
        "acceptance": _safe(acceptance),
        "measurement_provenance": _safe(provenance),
        "tool_friction": _safe(friction),
        "odyssey_wall": _safe(wall),
    }


def _line(label: str, value: Any) -> str:
    return f"  {label:34s} {value}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="machine-readable")
    args = ap.parse_args()
    doc = report()
    if args.json:
        print(json.dumps(doc, indent=1, sort_keys=True))
        return 0

    r = doc["roadmap"]
    print("ROADMAP")
    if "unavailable" in r:
        print(_line("unavailable", r["unavailable"]))
    else:
        print(_line("fresh against HEAD", r["fresh"]))
        print(_line("built / accepted", f"{r['built']} / {r['accepted']}"))
        print(_line("unresolved gates", r["unresolved"]))
        for cls, n in sorted(r["blocker_census"].items(), key=lambda kv: -kv[1]):
            print(_line(f"  {cls.lower()}", n))

    a = doc["acceptance"]
    print("\nACCEPTANCE INTEGRITY")
    if "unavailable" in a:
        print(_line("unavailable", a["unavailable"]))
    else:
        for k, v in sorted(a["criterion_recheck"].items()):
            print(_line(f"criterion {k.lower()}", v))
        if a["unverifiable_gates"]:
            print(_line("no re-checkable criterion", ", ".join(a["unverifiable_gates"][:4]) + " ..."))

    p = doc["measurement_provenance"]
    print("\nMEASUREMENT PROVENANCE")
    if "unavailable" in p:
        print(_line("unavailable", p["unavailable"]))
    else:
        print(_line("receipts with a hardware number", p["receipts_carrying_a_hardware_number"]))
        print(_line("  with provenance", p["with_provenance"]))
        print(_line("  WITHOUT", p["without_provenance"]))

    f = doc["tool_friction"]
    print("\nTOOL FRICTION")
    if "unavailable" in f:
        print(_line("unavailable", f["unavailable"]))
    else:
        print(_line("tool calls / failed", f"{f['tool_calls']} / {f['failed']} "
                                          f"({f['failure_rate']:.1%})"))
        print(_line("execution share of wall", f"{f['tool_execution_share_of_wall']:.2%}"))
        fnf = f["file_not_found"]
        print(_line("file-not-found recoverable", fnf["recoverable_wrong_directory_or_near_name"]))
        print(_line("model wall lost (pro rata)", f"{f['cost']['pro_rata_s']:.0f}s"))

    w = doc["odyssey_wall"]
    print("\nODYSSEY WALL")
    if "unavailable" in w:
        print(_line("unavailable", w["unavailable"]))
    else:
        print(_line("serial critical path", f"{w['on_serial_critical_path']} components"))
        print(_line("  of those recordable", f"{w['of_those_recordable']} "
                                             f"({w['serial_coverage']:.0%})"))
        print(_line("projection", w["projection"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
