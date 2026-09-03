"""Where would an Odyssey campaign's hours go, and what has to be true to go faster.

This is an accounting STRUCTURE with named holes, not a projection. It cannot be a
projection: receipts/future/ODYSSEY_WALL_INSTRUMENTATION.json records that the
substrate has never measured a duration -- 9,573 compile-economics events, every
wall_s exactly 0.0, and a run log whose only `timing` field is the boolean False.

So the useful output is not a number of hours. It is three things a number cannot
give:

    1. WHICH components exist, taken from the substrate's own event vocabulary
       rather than invented, and which of them the substrate can even record today.
    2. What each target rung REQUIRES -- a budget implied by arithmetic on the
       target, which is a requirement, never a prediction.
    3. WHICH component to instrument first, ranked by whether measuring it could
       change the answer to "is this rung reachable", not by how big we guess it is.

A component with no measurement contributes UNRECORDED, never zero. Zero is a
measurement that something took no time; that distinction is exactly the one the
compile-economics ledger lost.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tools.future import decision_value

REPO = Path(__file__).resolve().parents[2]
ECONOMICS = REPO / "workspace" / "campaign" / "odyssey" / "COMPILE_ECONOMICS.jsonl"

#: Evidence states for one component's duration. UNRECORDED is not zero.
MEASURED = "MEASURED"
UNRECORDED = "UNRECORDED"
NOT_INSTRUMENTED = "NOT_INSTRUMENTED"   # the substrate has no event for this at all

SECONDS_PER_HOUR = 3600.0


class UnrecordedWall(ValueError):
    """Raised when a projected wall is asked for and no component is measured."""


@dataclass(frozen=True)
class Component:
    """One place Odyssey wall can go."""
    name: str
    category: str
    #: The substrate's own event key, when one exists. None means nothing records it.
    substrate_event: str | None
    #: True when the work can run concurrently with other components by construction
    #: (a launched subprocess), False when a later step consumes its output.
    parallel_by_construction: bool | None
    #: Does its duration scale with the resident model's throughput?
    resident_dependent: bool
    #: Could a previous run's result be reused instead of recomputed?
    cacheable: bool
    note: str = ""

    @property
    def evidence(self) -> str:
        return NOT_INSTRUMENTED if self.substrate_event is None else UNRECORDED

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "substrate_event": self.substrate_event,
            "evidence": self.evidence,
            "measured_seconds": None,
            "parallel_by_construction": self.parallel_by_construction,
            "resident_dependent": self.resident_dependent,
            "cacheable": self.cacheable,
            "note": self.note,
        }


def _c(name, category, event, parallel, resident, cacheable, note=""):
    return Component(name, category, event, parallel, resident, cacheable, note)


#: The decomposition. Every entry with a substrate_event is taken from
#: tools/odyssey_costmodel.EVENT_WALL, which is the substrate's own vocabulary --
#: not a category list invented for this model. Entries with substrate_event=None
#: are places wall demonstrably goes that NOTHING in the substrate can record,
#: which is the more useful half of this table.
COMPONENTS: tuple[Component, ...] = (
    # --- the substrate has an event key for these -------------------------
    _c("acquisition", "artifact", "acquisition_wall", True, False, True,
       "fetching a specimen; bounded by network, reusable across campaigns"),
    _c("census", "artifact", "census_wall", True, False, True,
       "indexing a specimen's tensors; deterministic given the bytes"),
    _c("doctor_fast", "diagnosis", "doctor_fast_wall", True, False, True),
    _c("doctor_full", "diagnosis", "doctor_full_wall", True, False, True),
    _c("gravity_search", "compilation", "gravity_search_wall", True, False, False,
       "the search itself; the largest declared compile term"),
    _c("gravity_pack", "compilation", "gravity_pack_wall", True, False, False),
    _c("gravity_verify", "verification", "gravity_verify_wall", True, False, False),
    _c("nx_lower", "compilation", "nx_lower_wall", True, False, False),
    _c("kernel_probe", "measurement", "kernel_probe_wall", False, False, False,
       "needs an exclusive GPU lane; cannot overlap another GPU measurement"),
    _c("gpu_total", "measurement", "total_gpu_wall", False, False, False,
       "every GPU-exclusive measurement, serialised against itself"),
    _c("cpu_total", "deterministic_tools", "total_cpu_wall", True, False, True),
    _c("retirement", "artifact", "retirement_wall", True, False, False),
    # --- nothing in the substrate records these ---------------------------
    _c("resident_prefill", "cognition", None, False, True, True,
       "prompt/context prefill per turn; the stable-prefix reuse target"),
    _c("resident_decode", "cognition", None, False, True, False,
       "reasoning and generation; the irreducible cognition this campaign protects"),
    _c("tool_call_recovery", "friction", None, False, True, False,
       "malformed call, schema repair, re-probe -- each costs a full resident turn"),
    _c("context_reconstruction", "friction", None, False, True, True,
       "rebuilding what the resident already knew; HCLI-owned, consumed not built here"),
    _c("receipt_generation", "bookkeeping", None, True, True, False,
       "resident cognition spent writing evidence boilerplate"),
    _c("result_ingestion", "bookkeeping", None, True, True, True,
       "copying numbers between files; deterministic and migratable off the resident"),
    _c("scheduler_idle", "coordination", None, False, False, False,
       "gaps where a lane waits with nothing admitted; pure loss"),
    _c("dependency_wait", "coordination", None, False, False, False,
       "a lane blocked on another lane's output; the serial critical path"),
    _c("failed_hypotheses", "science", None, False, True, False,
       "experiments that produce a negative result; useful, not waste"),
    _c("duplicated_work", "coordination", None, True, True, True,
       "two lanes deriving the same fact; pure loss"),
)

CATEGORIES = tuple(dict.fromkeys(c.category for c in COMPONENTS))


def instrumented() -> tuple[Component, ...]:
    return tuple(c for c in COMPONENTS if c.substrate_event is not None)


def uninstrumented() -> tuple[Component, ...]:
    return tuple(c for c in COMPONENTS if c.substrate_event is None)


def measured_components() -> tuple[Component, ...]:
    """Components with a real duration on disk. Currently none, and that is the point."""
    if not ECONOMICS.is_file():
        return ()
    events_with_wall = set()
    for line in ECONOMICS.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if float(row.get("wall_s") or 0.0) > 0:
            events_with_wall.add(str(row.get("event")))
    return tuple(c for c in COMPONENTS
                 if c.substrate_event and str(c.substrate_event).split("_")[0] in events_with_wall)


def projected_wall_hours() -> float:
    """Refuses. A campaign wall with no measured component is a number about nothing."""
    have = measured_components()
    if not have:
        raise UnrecordedWall(
            "no Odyssey component has a measured duration: COMPILE_ECONOMICS.jsonl "
            f"holds events whose wall_s is uniformly 0.0, and {len(uninstrumented())} "
            "of the components below have no substrate event at all. A projected "
            "campaign wall would be arithmetic over nothing. Instrument first; see "
            "instrumentation_order()."
        )
    raise UnrecordedWall(
        f"{len(have)} of {len(COMPONENTS)} components are measured. A projection "
        "needs the serial critical path, not a subset of the terms."
    )


def requirements_for(target_hours: float, *, serial_fraction: float) -> dict[str, Any]:
    """What a rung REQUIRES. Arithmetic on the target, never a prediction.

    Amdahl bounds the wall from below by the part that cannot be made concurrent:
    with total work T and serial fraction s, no amount of parallelism goes below
    s*T. So a target H implies a ceiling on total work, T <= H/s, and that ceiling
    is a REQUIREMENT the campaign must satisfy -- it says nothing about whether it
    does. serial_fraction is itself unmeasured; sweep it rather than picking one.
    """
    if not 0 < serial_fraction <= 1:
        raise ValueError(f"serial_fraction must be in (0, 1], got {serial_fraction}")
    if target_hours <= 0:
        raise ValueError(f"target_hours must be positive, got {target_hours}")
    return {
        "target_hours": target_hours,
        "serial_fraction": serial_fraction,
        "max_total_work_hours": target_hours / serial_fraction,
        "irreducible_serial_hours_at_that_work": target_hours,
        "requirement": (
            f"total campaign work must not exceed {target_hours / serial_fraction:.1f} h "
            f"if {serial_fraction:.0%} of it is serial, because no concurrency goes "
            f"below the serial part"
        ),
        "is_a_prediction": False,
    }


LADDER_HOURS = (48.0, 36.0, 24.0, 18.0, 12.0)
SERIAL_SWEEP = (0.05, 0.1, 0.2, 0.3, 0.5, 0.8)


def ladder() -> list[dict[str, Any]]:
    """Every rung against every serial fraction. A requirement table, not a forecast."""
    return [requirements_for(h, serial_fraction=s)
            for h in LADDER_HOURS for s in SERIAL_SWEEP]


def instrumentation_order() -> dict[str, Any]:
    """Which component to instrument first, ranked by STRUCTURE, not by guessed size.

    The obvious design -- sweep each category's plausible share of a nominal 48 h
    and rank by which can move the ladder rung -- was tried and produces nothing.
    Applying one uniform share range to every category makes all eleven reach
    exactly the same rungs, so the ranking is arbitrary. That is the honest
    answer to a question asked with no information in it: NOTHING IS KNOWN ABOUT
    RELATIVE SIZE, so relative size cannot order the work.

    What IS known is structural, and it orders the work perfectly well:

      on the serial critical path   its wall lands on the campaign wall directly,
                                    where a concurrent component's may not
      not instrumented at all       it cannot be measured even in principle today,
                                    so instrumenting it is a prerequisite, not a task
      cacheable                     it could be deleted rather than shortened, and
                                    measuring it prices that deletion
      resident dependent            its wall moves when the resident model changes,
                                    so it must be normalised before any rung is claimed

    A component that is serial, uninstrumented and not cacheable is the worst
    case: it lands on the critical path, cannot be removed, and cannot currently
    be seen at all.
    """
    def score(c: Component) -> tuple[int, int, int, int]:
        return (
            1 if c.parallel_by_construction is False else 0,   # serial: on the path
            1 if c.substrate_event is None else 0,             # invisible today
            0 if c.cacheable else 1,                           # cannot be deleted
            1 if c.resident_dependent else 0,                  # needs normalisation
        )

    ranked = sorted(COMPONENTS, key=lambda c: (score(c), c.name), reverse=True)
    rows = []
    for i, c in enumerate(ranked, 1):
        serial, invisible, irreducible, resident = score(c)
        rows.append({
            "instrument_order": i,
            "component": c.name,
            "category": c.category,
            "on_serial_critical_path": bool(serial),
            "invisible_to_the_substrate_today": bool(invisible),
            "cannot_be_deleted_only_shortened": bool(irreducible),
            "moves_with_the_resident_model": bool(resident),
            "why": _why(serial, invisible, irreducible, resident),
        })
    return {
        "ordering_basis": "structural, not size",
        "why_not_size": (
            "sweeping each category's plausible share of a nominal 48 h and ranking "
            "by which can change the ladder rung was tried: with one uniform share "
            "range per category, all eleven reach the same rungs and the ordering is "
            "arbitrary. No size is known, so size cannot order the work."
        ),
        "worst_case_shape": "serial + invisible + not cacheable",
        "rows": rows,
    }


def _why(serial: int, invisible: int, irreducible: int, resident: int) -> str:
    bits = []
    if serial:
        bits.append("lands directly on the campaign wall")
    else:
        bits.append("can overlap other work")
    if invisible:
        bits.append("nothing records it, so instrumenting it comes before measuring it")
    if not irreducible:
        bits.append("cacheable, so it could be deleted rather than shortened")
    if resident:
        bits.append("scales with the resident model and must be normalised")
    return "; ".join(bits)


def report() -> dict[str, Any]:
    """The whole accounting structure, machine-readable."""
    have = measured_components()
    try:
        projected: Any = projected_wall_hours()
    except UnrecordedWall as exc:
        projected = {"refused": str(exc)}
    return {
        "schema": "hawking.future.odyssey_wall.v1",
        "evidence_tier": "STATIC",
        "odyssey_launched": False,
        "components": [c.as_dict() for c in COMPONENTS],
        "categories": list(CATEGORIES),
        "counts": {
            "components": len(COMPONENTS),
            "with_a_substrate_event": len(instrumented()),
            "with_no_substrate_event": len(uninstrumented()),
            "measured": len(have),
        },
        "projected_campaign_wall": projected,
        "ladder_requirements": ladder(),
        "instrumentation_order": instrumentation_order(),
        "claim_boundary": (
            "STATIC accounting structure. Every component is UNRECORDED or "
            "NOT_INSTRUMENTED; none carries a duration. The ladder table states what "
            "each rung REQUIRES, which is arithmetic on the target and not a forecast. "
            "No Odyssey was launched and no campaign wall is projected."
        ),
    }


def build() -> Path:
    """Write the accounting structure as a receipt."""
    from tools.future._common import write_receipt
    return write_receipt("ODYSSEY_WALL_MODEL.json", report(),
                         recorded_by="tools/future/odyssey_wall.py")


def main() -> int:
    r = report()
    c = r["counts"]
    print(f"components {c['components']}: {c['with_a_substrate_event']} have a substrate "
          f"event, {c['with_no_substrate_event']} have none, {c['measured']} are measured")
    serial = [x for x in COMPONENTS if x.parallel_by_construction is False]
    rec = [x for x in serial if x.substrate_event is not None]
    print(f"serial critical path: {len(serial)} components, {len(rec)} recordable "
          f"({len(rec) / len(serial):.0%})")
    print(f"projected wall: {r['projected_campaign_wall']['refused'][:90]}...")
    print("instrument first:")
    for row in r["instrumentation_order"]["rows"][:5]:
        print(f"  {row['instrument_order']}. {row['component']}")
    print(f"wrote {build()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
