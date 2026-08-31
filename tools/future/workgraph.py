"""WORKGRAPH — graph scheduler with fourteen durable resource lanes.

HCLI already has a DAG dispatcher (`hcli/scheduler.py` + `hcli/workunit.py` +
`hcli/dag_store.py`). It is FIFO by ready_at, occupancy-capped on HCLI
ResourceClass, exclusive on GPU_EXCLUSIVE vs GPU_DECODE, and durable via
dag.json. This sidecar does not replace that dispatcher and does not execute.

What was missing, and what this module is: a graph whose ready set is computed
from dependency and verification edges; fourteen named resource lanes with
declared exclusivity and capacity; mutation-scope and exclusive-lane
co-schedule refusal; expected-information-per-cost selection with starvation
reporting; SLEEPING units for unqualified hardware (never a synthetic result);
and a disk document the resident can reload after process death.

    python3 tools/future/workgraph.py --build
    python3 tools/future/workgraph.py --selftest
    python3 tools/future/workgraph.py --tick
    python3 -m pytest tools/future/test_workgraph.py -q

Everything emitted is STATIC_ONLY, bench UNKNOWN, gpu_authority false.
This module produces neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE.
It never takes a GPU lease, never starts a resident, never calls a network.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO

import argparse
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from hcli.persist import atomic_write_json
from hcli.workunit import WorkUnit
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, git

RECEIPT = "WORKGRAPH_STATE.json"
SCHEMA = "hawking.future.workgraph.v1"
VERSION = 1
RECORDED_BY = "tools/future/workgraph.py"
DURABLE_SCHEMA = "hawking.future.workgraph.durable.v1"
DURABLE_NAME = "workgraph.json"
DEFAULT_WORKSPACE = RECEIPTS / "workgraph_workspace"

ERAS = (
    "I Genesis of the Laboratory",
    "II Compounding Civilization",
    "III Autonomous Science Civilization",
    "IV Synthetic Machine Civilization",
    "V Released Hawking Civilization",
)
ODYSSEYS = (
    "I WHAT IS TRUE?",
    "II WHAT DID HAWKING ALREADY LEARN?",
    "III WHERE IS HAWKING WRONG?",
)

# Fourteen resource lanes. Order is the contract order; exclusivity and
# capacity are declared policy, not a hardware measurement.
LANE_IDS: tuple[str, ...] = (
    "GPU_PROTECTED",
    "GPU_DIAGNOSTIC",
    "CPU_BUILD",
    "CPU_REPRESENTATION",
    "CPU_ANALYSIS",
    "CPU_VERIFY",
    "DISK_IO",
    "NETWORK_RESEARCH",
    "MODELLAKE",
    "FPGA_SIM",
    "ANE",
    "LOCAL_MODEL",
    "STRONG_MODEL",
    "GROK_SWARM",
)

# HCLI ResourceClass projection so emitted units round-trip through WorkUnit.
# Recovered from hcli/resources.py ResourceClass and workunit_species.REPAT_RESOURCE.
LANE_TO_HCLI: dict[str, str] = {
    "GPU_PROTECTED": "GPU_EXCLUSIVE",
    "GPU_DIAGNOSTIC": "GPU_DECODE",
    "CPU_BUILD": "COMPILE",
    "CPU_REPRESENTATION": "MEMORY_HEAVY",
    "CPU_ANALYSIS": "STATIC_ANALYSIS",
    "CPU_VERIFY": "TEST",
    "DISK_IO": "IO_HEAVY",
    "NETWORK_RESEARCH": "TOOL_WAIT",
    "MODELLAKE": "IO_HEAVY",
    "FPGA_SIM": "COMPILE",
    "ANE": "GPU_EXCLUSIVE",
    "LOCAL_MODEL": "CPU_HEAVY",
    "STRONG_MODEL": "LIGHT_CONTROL",
    "GROK_SWARM": "GROK",
}

HCLI_TO_LANE: dict[str, str] = {
    "GPU_EXCLUSIVE": "GPU_PROTECTED",
    "GPU_DECODE": "GPU_DIAGNOSTIC",
    "GPU_DIRTY_OK": "GPU_DIAGNOSTIC",
    "CPU_HEAVY": "CPU_ANALYSIS",
    "COMPILE": "CPU_BUILD",
    "TEST": "CPU_VERIFY",
    "TEST_AUTHORING": "CPU_VERIFY",
    "STATIC_ANALYSIS": "CPU_ANALYSIS",
    "MEMORY_HEAVY": "CPU_REPRESENTATION",
    "IO_HEAVY": "DISK_IO",
    "TOOL_WAIT": "NETWORK_RESEARCH",
    "LIGHT_CONTROL": "CPU_ANALYSIS",
    "GROK": "GROK_SWARM",
    "MUTATION": "CPU_BUILD",
}

SPECIES_LANE: dict[str, str] = {
    "accelerator_candidate_qualification": "GPU_PROTECTED",
    "architecture_transfer": "GPU_PROTECTED",
    "odyssey_ii_transfer_experiment": "CPU_ANALYSIS",
    "odyssey_iii_adversarial_experiment": "CPU_ANALYSIS",
    "fpga_simulation": "FPGA_SIM",
    "hardware_doctor_experiment": "CPU_ANALYSIS",
    "learned_compiler_experiment": "CPU_REPRESENTATION",
    "fusion_simulation": "FPGA_SIM",
    "independent_reproduction": "CPU_VERIFY",
    "green_machine_measurement": "CPU_ANALYSIS",
}

HARDWARE_LANES = frozenset({"GPU_PROTECTED", "GPU_DIAGNOSTIC", "ANE"})

# Ranking proxies. Same integer rule as hardware_doctor.rank_queue and
# resident_optimizer.rank_hypotheses: -(info * 60 // cost). Not a measurement.
INFO_HIGH, INFO_MEDIUM, INFO_LOW = 3, 2, 1
STARVATION_THRESHOLD = 8

REQUIRED_FIELDS: tuple[str, ...] = (
    "id",
    "role",
    "description",
    "dependencies",
    "resource_lane",
    "mutation_scope",
    "verifier",
    "expected_information_gain",
    "cost_units",
)

GRAPH_STATUSES = frozenset(
    {"pending", "ready", "running", "completed", "failed", "sleeping"}
)

CLAIM_BOUNDARY = (
    "Static sidecar artifact. WorkGraph emits a schedule. It does not execute, "
    "does not take a GPU lease, does not promote, and does not raise evidence "
    "class above STATIC_ONLY. SLEEPING hardware work is not a result."
)

# Recovered from the live qualification pipeline receipt and Codex's blocker
# list. Recorded as data; this module does not probe Metal, xcrun, or flock.
PHYSICAL_BLOCKERS: tuple[str, ...] = (
    "MetalContext reports NO Metal-capable GPU on this host",
    "xcrun cannot locate the Metal compiler under CommandLineTools",
    "protected bench lock files exist; holder pids unproven; flock would be a seizure",
    "qualification pipeline classifies the machine HEAVY and will not quiesce standing workers",
    "Flash source-independent NX is SCAFFOLD_ONLY, not qualified",
    "teacher capture is 0/256",
)

EVIDENCE_QUEUE = (
    RECEIPTS / "evidence" / "ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json"
)
WORKUNITS_RECEIPT = RECEIPTS / "HCLI_FUTURE_WORKUNITS.json"
QUAL_PIPELINE_RECEIPT = RECEIPTS / "QUALIFICATION_PIPELINE.json"
FRONTIER_RECEIPT = RECEIPTS / "CLAUDE_GLOBAL_FRONTIER.json"

# HCLI grok fallback (hcli/resources.py ResourceLimits.grok default). Policy, not a bench.
GROK_FALLBACK_CAPACITY = 2


# ---------------------------------------------------------------------------
# Errors. A guard nobody has watched fail is not a guard.
# ---------------------------------------------------------------------------


class WorkGraphError(ValueError):
    """Base error for the graph scheduler."""


class AdmissionError(WorkGraphError):
    """A unit missing a required field, or carrying an illegal field, is REJECTED."""

    def __init__(self, reason: str, *, missing: Sequence[str] = ()) -> None:
        self.reason = reason
        self.missing = list(missing)
        super().__init__(reason)


class IdentityConflict(WorkGraphError):
    """Same id, different graph content. Never a silent overwrite."""


class CycleError(AdmissionError):
    """Admitting this unit would create a dependency cycle."""


class ExecutionRefused(WorkGraphError):
    """WorkGraph does not execute. Execution is a sibling lane's concern."""


class SyntheticResultError(WorkGraphError):
    """SLEEPING or never-scheduled work cannot be completed with a fabricated result."""


class DurableCorruptError(WorkGraphError):
    """The durable graph document exists but is not a valid graph."""


# ---------------------------------------------------------------------------
# Lanes
# ---------------------------------------------------------------------------


def _ncpu() -> int:
    return os.cpu_count() or 1


@dataclass(frozen=True)
class LaneSpec:
    """Declared occupancy policy for one resource lane."""

    id: str
    exclusive: bool
    capacity: int
    physical: str | None
    hcli_resource_class: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "exclusive": self.exclusive,
            "capacity": int(self.capacity),
            "physical": self.physical,
            "hcli_resource_class": self.hcli_resource_class,
            "note": self.note,
        }


def lane_specs(*, ncpu: int | None = None) -> dict[str, LaneSpec]:
    """Fourteen lanes. Capacity is occupancy policy derived from cpu_count / HCLI fallback."""
    n = int(ncpu) if ncpu is not None else _ncpu()
    n = max(1, n)
    io_cap = max(8, n)
    rows = (
        LaneSpec(
            "GPU_PROTECTED",
            True,
            1,
            "metal_gpu",
            LANE_TO_HCLI["GPU_PROTECTED"],
            "exclusive protected GPU window; Codex-owned execution; sidecar schedules only",
        ),
        LaneSpec(
            "GPU_DIAGNOSTIC",
            True,
            1,
            "metal_gpu",
            LANE_TO_HCLI["GPU_DIAGNOSTIC"],
            "exclusive diagnostic GPU; mutually exclusive with GPU_PROTECTED via physical=metal_gpu",
        ),
        LaneSpec(
            "CPU_BUILD",
            False,
            n,
            None,
            LANE_TO_HCLI["CPU_BUILD"],
            "compile / build; not exclusive",
        ),
        LaneSpec(
            "CPU_REPRESENTATION",
            False,
            n,
            None,
            LANE_TO_HCLI["CPU_REPRESENTATION"],
            "representation / LPC contract work; not exclusive",
        ),
        LaneSpec(
            "CPU_ANALYSIS",
            False,
            n,
            None,
            LANE_TO_HCLI["CPU_ANALYSIS"],
            "static analysis; explicitly not exclusive",
        ),
        LaneSpec(
            "CPU_VERIFY",
            False,
            n,
            None,
            LANE_TO_HCLI["CPU_VERIFY"],
            "independent reproduction / tests; not exclusive",
        ),
        LaneSpec(
            "DISK_IO",
            False,
            io_cap,
            None,
            LANE_TO_HCLI["DISK_IO"],
            "receipt and corpus IO; not exclusive",
        ),
        LaneSpec(
            "NETWORK_RESEARCH",
            False,
            io_cap,
            None,
            LANE_TO_HCLI["NETWORK_RESEARCH"],
            "read-only research; not exclusive; this sidecar itself never opens a socket",
        ),
        LaneSpec(
            "MODELLAKE",
            True,
            1,
            None,
            LANE_TO_HCLI["MODELLAKE"],
            "single lake operator",
        ),
        LaneSpec(
            "FPGA_SIM",
            False,
            n,
            None,
            LANE_TO_HCLI["FPGA_SIM"],
            "CPU FPGA simulation; FPGA belongs to Accelerator/Physical Compiler/Fusion",
        ),
        LaneSpec(
            "ANE",
            True,
            1,
            "ane",
            LANE_TO_HCLI["ANE"],
            "exclusive ANE; sleeping until a public Core ML path exists",
        ),
        LaneSpec(
            "LOCAL_MODEL",
            True,
            1,
            None,
            LANE_TO_HCLI["LOCAL_MODEL"],
            "single local-resident slot",
        ),
        LaneSpec(
            "STRONG_MODEL",
            True,
            1,
            None,
            LANE_TO_HCLI["STRONG_MODEL"],
            "single strong-model slot",
        ),
        LaneSpec(
            "GROK_SWARM",
            False,
            GROK_FALLBACK_CAPACITY,
            None,
            LANE_TO_HCLI["GROK_SWARM"],
            "HCLI ResourceLimits.grok fallback capacity=2; not a measurement",
        ),
    )
    specs = {row.id: row for row in rows}
    if tuple(specs) != LANE_IDS:
        raise WorkGraphError("lane_specs() must declare exactly LANE_IDS in contract order")
    return specs


# ---------------------------------------------------------------------------
# Hardware qualification — receipts only, never a probe
# ---------------------------------------------------------------------------


def hardware_qualification() -> dict[str, Any]:
    """This sidecar has no GPU authority. Physical GPU/ANE work SLEEPS.

    Reads the qualification-pipeline receipt when present so the reasons are
    disk-backed. Absence of that receipt is not evidence the pipeline does
    not exist; reasons fall back to the recovered blocker list.
    """
    pipeline: dict[str, Any] | None = None
    pipeline_path: str | None = None
    if QUAL_PIPELINE_RECEIPT.is_file():
        try:
            pipeline = load_json(QUAL_PIPELINE_RECEIPT)
            pipeline_path = str(QUAL_PIPELINE_RECEIPT.relative_to(REPO))
        except (OSError, json.JSONDecodeError, ValueError):
            pipeline = None
    contamination = None
    lease_present = False
    if isinstance(pipeline, dict):
        body = pipeline.get("pipeline") if isinstance(pipeline.get("pipeline"), dict) else pipeline
        contamination = body.get("contamination_class")
        lease_present = bool(body.get("lease_present"))
    reasons = list(PHYSICAL_BLOCKERS)
    if contamination is not None:
        reasons.append(
            f"QUALIFICATION_PIPELINE.json pipeline.contamination_class={contamination!r}"
        )
    if pipeline_path is not None:
        reasons.append(f"lease_present={lease_present} from {pipeline_path}")
    return {
        "qualified": False,
        "gpu_authority": False,
        "lease_present": bool(lease_present),
        "contamination_class": contamination,
        "pipeline_receipt": pipeline_path,
        "pipeline_receipt_present": pipeline_path is not None,
        "reasons": reasons,
        "rule": (
            "blocked physical work becomes a SLEEPING WorkUnit; it never "
            "becomes a synthetic result. UNKNOWN is the correct hardware answer."
        ),
    }


def hardware_is_qualified() -> bool:
    return bool(hardware_qualification()["qualified"])


# ---------------------------------------------------------------------------
# Unit construction / admission
# ---------------------------------------------------------------------------


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


def _as_str_tuple(value: Any, *, field: str) -> tuple[str, ...]:
    if value is None:
        raise AdmissionError(f"{field} is required", missing=[field])
    if isinstance(value, (str, bytes)):
        raise AdmissionError(f"{field} must be a list of strings, not {type(value).__name__}")
    if not isinstance(value, (list, tuple)):
        raise AdmissionError(f"{field} must be a list of strings")
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return tuple(out)


def graph_identity(unit: Mapping[str, Any]) -> str:
    payload = {
        "role": str(unit.get("role") or ""),
        "description": str(unit.get("description") or ""),
        "dependencies": [str(d) for d in (unit.get("dependencies") or [])],
        "resource_lane": str(unit.get("resource_lane") or ""),
        "verifier": str(unit.get("verifier") or ""),
        "mutation_scope": [str(s) for s in (unit.get("mutation_scope") or [])],
        "verification_depends_on": [str(s) for s in (unit.get("verification_depends_on") or [])],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def make_unit(
    *,
    id: str,
    role: str,
    description: str,
    dependencies: Sequence[str],
    resource_lane: str,
    mutation_scope: Sequence[str],
    verifier: str,
    expected_information_gain: int,
    cost_units: int,
    verification_depends_on: Sequence[str] = (),
    requires_hardware: bool | None = None,
    species: str | None = None,
    effect_class: str = "READ_ONLY",
    claim_boundary: str = CLAIM_BOUNDARY,
    classification: str | None = None,
    blocked_reason: str | None = None,
    extras: Mapping[str, Any] | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    """Construct one graph unit or RAISE. Missing required fields are REJECTED."""
    raw = {
        "id": id,
        "role": role,
        "description": description,
        "dependencies": dependencies,
        "resource_lane": resource_lane,
        "mutation_scope": mutation_scope,
        "verifier": verifier,
        "expected_information_gain": expected_information_gain,
        "cost_units": cost_units,
    }
    missing = [name for name in REQUIRED_FIELDS if name not in raw or not _field_supplied(name, raw[name])]
    if missing:
        raise AdmissionError(
            f"{id!r}: missing required field(s) {missing}",
            missing=missing,
        )
    lane = str(resource_lane).strip()
    if lane not in LANE_IDS:
        raise AdmissionError(f"{id!r}: resource_lane {resource_lane!r} is not one of the fourteen lanes")
    if not str(verifier).strip() or str(verifier).strip().lower() in {"self", "none", "disable", "weaken"}:
        raise AdmissionError(f"{id!r}: verifier {verifier!r} would weaken verification")
    try:
        info = int(expected_information_gain)
        cost = int(cost_units)
    except (TypeError, ValueError) as exc:
        raise AdmissionError(f"{id!r}: expected_information_gain and cost_units must be integers") from exc
    if info not in {INFO_LOW, INFO_MEDIUM, INFO_HIGH}:
        raise AdmissionError(f"{id!r}: expected_information_gain must be 1, 2, or 3; got {info}")
    if cost < 1:
        raise AdmissionError(f"{id!r}: cost_units must be >= 1")

    deps = _as_str_tuple(dependencies, field="dependencies")
    scope = tuple(sorted(_as_str_tuple(mutation_scope, field="mutation_scope")))
    vdeps = _as_str_tuple(verification_depends_on, field="verification_depends_on")
    if str(id) in deps or str(id) in vdeps:
        raise CycleError(f"{id!r}: unit cannot depend on itself")

    hardware = bool(requires_hardware) if requires_hardware is not None else lane in HARDWARE_LANES
    if status is None:
        if hardware and not hardware_is_qualified():
            status = "sleeping"
        else:
            status = "pending"
    if status not in GRAPH_STATUSES:
        raise AdmissionError(f"{id!r}: status {status!r} is not a graph status")
    if status == "sleeping" and not hardware:
        raise AdmissionError(f"{id!r}: SLEEPING is reserved for hardware-gated work")

    blocked = blocked_reason
    if status == "sleeping" and not blocked:
        blocked = (
            "hardware unqualified on this sidecar: GPU/ANE work sleeps until "
            "Codex qualifies the host; never a synthetic result"
        )

    unit: dict[str, Any] = {
        "id": str(id),
        "role": str(role),
        "description": str(description),
        "dependencies": list(deps),
        "resource_lane": lane,
        "mutation_scope": list(scope),
        "verifier": str(verifier),
        "expected_information_gain": info,
        "cost_units": cost,
        "verification_depends_on": list(vdeps),
        "requires_hardware": hardware,
        "status": status,
        "skipped_ticks": 0,
        "assigned_tick": None,
        "completed_tick": None,
        "hcli_resource_class": LANE_TO_HCLI[lane],
        "species": species,
        "effect_class": str(effect_class or "READ_ONLY"),
        "claim_boundary": str(claim_boundary or CLAIM_BOUNDARY),
        "classification": classification if classification is not None else (
            "SLEEPING" if status == "sleeping" else "STATIC_ONLY"
        ),
        "blocked_reason": blocked,
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "gpu_authority": False,
        "content_hash": "",
    }
    if extras:
        for key, value in extras.items():
            if key in HARDWARE_FIELDS and isinstance(value, (int, float)):
                raise AdmissionError(
                    f"{id!r}: extras.{key} is a hardware number; sidecar has no GPU authority"
                )
            if key not in unit or unit[key] in (None, "", []):
                unit[key] = value
    unit["content_hash"] = graph_identity(unit)
    return unit


def _field_supplied(name: str, value: Any) -> bool:
    if name in {"dependencies", "mutation_scope"}:
        return value is not None
    return _present(value)


def _assert_no_cycle(units: Mapping[str, Mapping[str, Any]], new_id: str, edges: Sequence[str]) -> None:
    graph: dict[str, list[str]] = {
        uid: list(u.get("dependencies") or []) + list(u.get("verification_depends_on") or [])
        for uid, u in units.items()
    }
    graph[new_id] = list(edges)
    visiting: set[str] = set()
    seen: set[str] = set()

    def dfs(node: str) -> bool:
        if node in visiting:
            return True
        if node in seen:
            return False
        visiting.add(node)
        for nxt in graph.get(node, ()):
            if nxt not in graph and nxt != node:
                continue
            if dfs(nxt):
                return True
        visiting.remove(node)
        seen.add(node)
        return False

    if dfs(new_id):
        raise CycleError(f"{new_id!r}: dependency/verification cycle")


# ---------------------------------------------------------------------------
# Ready set, conflicts, selection
# ---------------------------------------------------------------------------


def _deps_satisfied(unit: Mapping[str, Any], units: Mapping[str, Mapping[str, Any]]) -> bool:
    needed = list(unit.get("dependencies") or []) + list(unit.get("verification_depends_on") or [])
    for dep_id in needed:
        dep = units.get(dep_id)
        if dep is None:
            return False
        if dep.get("status") != "completed":
            return False
    return True


def scopes_intersect(a: Sequence[str], b: Sequence[str]) -> bool:
    if not a or not b:
        return False
    return bool(set(a) & set(b))


def information_key(unit: Mapping[str, Any]) -> tuple[Any, ...]:
    """Prefer expected information gain per unit cost; ties by cost then id.

    Integer key -(info*60//cost) is the hardware_doctor / resident_optimizer
    rule. skipped_ticks past STARVATION_THRESHOLD boost so a lane cannot be
    starved indefinitely.
    """
    info = int(unit["expected_information_gain"])
    cost = int(unit["cost_units"])
    skipped = int(unit.get("skipped_ticks") or 0)
    boost = skipped if skipped >= STARVATION_THRESHOLD else 0
    return (-boost, -(info * 60 // cost), cost, str(unit["id"]))


def select_order(ready: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(u) for u in sorted(ready, key=information_key)]


# ---------------------------------------------------------------------------
# WorkGraph
# ---------------------------------------------------------------------------


class WorkGraph:
    """Durable graph of WorkUnits. Emits a schedule. Does not execute."""

    def __init__(
        self,
        workspace: str | os.PathLike[str] | None = None,
        *,
        lanes: Mapping[str, LaneSpec] | None = None,
        ncpu: int | None = None,
    ) -> None:
        self.workspace = Path(workspace) if workspace is not None else None
        self.lanes = dict(lanes) if lanes is not None else lane_specs(ncpu=ncpu)
        if tuple(self.lanes) != LANE_IDS:
            raise WorkGraphError("WorkGraph.lanes must be the fourteen contract lanes")
        self.units: dict[str, dict[str, Any]] = {}
        self.rejected: list[dict[str, Any]] = []
        self.tick_n = 0
        self.last_schedule: dict[str, Any] | None = None
        self.starvation_reports: list[dict[str, Any]] = []
        self.resumed = False

    @property
    def path(self) -> Path | None:
        if self.workspace is None:
            return None
        return self.workspace / DURABLE_NAME

    def save(self) -> Path | None:
        if self.path is None:
            return None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.path, self.to_durable())
        return self.path

    def to_durable(self) -> dict[str, Any]:
        ready_ids = [u["id"] for u in self.compute_ready(mutate=False)]
        running_ids = self.ids_in("running")
        completed_ids = self.ids_in("completed")
        return {
            "schema": DURABLE_SCHEMA,
            "version": VERSION,
            "tick": self.tick_n,
            "units": {uid: dict(self.units[uid]) for uid in sorted(self.units)},
            "rejected": list(self.rejected),
            "starvation_reports": list(self.starvation_reports),
            "last_schedule": self.last_schedule,
            "snapshot": {
                "ready_ids": ready_ids,
                "running_ids": running_ids,
                "completed_ids": completed_ids,
                "sleeping_ids": self.ids_in("sleeping"),
                "failed_ids": self.ids_in("failed"),
            },
            "lanes": {lid: spec.to_dict() for lid, spec in self.lanes.items()},
            "gpu_authority": False,
            "evidence_class": "STATIC_ONLY",
            "bench_state": "UNKNOWN",
            "executes": False,
        }

    @classmethod
    def load(
        cls,
        workspace: str | os.PathLike[str],
        *,
        ncpu: int | None = None,
    ) -> "WorkGraph":
        root = Path(workspace)
        path = root / DURABLE_NAME if root.is_dir() or not root.suffix else root
        if path.suffix != ".json":
            path = root / DURABLE_NAME
            workspace_dir = root
        else:
            workspace_dir = path.parent
        if not path.is_file():
            g = cls(workspace=workspace_dir, ncpu=ncpu)
            g.resumed = False
            return g
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DurableCorruptError(f"{path} is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise DurableCorruptError(f"{path} root is not an object")
        if data.get("schema") not in {DURABLE_SCHEMA, SCHEMA}:
            raise DurableCorruptError(f"{path} schema {data.get('schema')!r} is not a workgraph document")
        units_blob = data.get("units")
        if not isinstance(units_blob, dict):
            raise DurableCorruptError(f"{path} missing units object")
        g = cls(workspace=workspace_dir, ncpu=ncpu)
        for uid, payload in units_blob.items():
            if not isinstance(payload, dict):
                raise DurableCorruptError(f"unit {uid!r} is not an object")
            g.units[str(uid)] = dict(payload)
            g.units[str(uid)]["id"] = str(uid)
        g.tick_n = int(data.get("tick") or 0)
        g.rejected = list(data.get("rejected") or [])
        g.starvation_reports = list(data.get("starvation_reports") or [])
        ld = data.get("last_schedule")
        g.last_schedule = dict(ld) if isinstance(ld, dict) else None
        g.resumed = True
        g.compute_ready(mutate=True)
        return g

    def ids_in(self, status: str) -> list[str]:
        return sorted(uid for uid, u in self.units.items() if u.get("status") == status)

    def admit(self, unit: Mapping[str, Any] | None = None, **fields: Any) -> dict[str, Any]:
        """Admit one unit. Missing required fields -> REJECTED (not inserted)."""
        try:
            if unit is None:
                built = make_unit(**fields)
            elif isinstance(unit, dict) and fields:
                raise AdmissionError("admit() takes a unit dict or fields, not both")
            elif isinstance(unit, dict) and all(k in unit for k in REQUIRED_FIELDS):
                # Already constructed (or a raw dict that still needs make_unit).
                if "content_hash" in unit and "resource_lane" in unit and "status" in unit:
                    built = dict(unit)
                    missing = [
                        name
                        for name in REQUIRED_FIELDS
                        if name not in built or not _field_supplied(name, built[name])
                    ]
                    if missing:
                        raise AdmissionError(
                            f"{built.get('id')!r}: missing required field(s) {missing}",
                            missing=missing,
                        )
                    if built.get("resource_lane") not in LANE_IDS:
                        raise AdmissionError(
                            f"{built.get('id')!r}: resource_lane {built.get('resource_lane')!r} "
                            "is not one of the fourteen lanes"
                        )
                    built["content_hash"] = graph_identity(built)
                else:
                    built = make_unit(**{k: unit[k] for k in REQUIRED_FIELDS}, **{
                        k: unit[k]
                        for k in (
                            "verification_depends_on",
                            "requires_hardware",
                            "species",
                            "effect_class",
                            "claim_boundary",
                            "classification",
                            "blocked_reason",
                            "status",
                        )
                        if k in unit
                    })
            else:
                raw = dict(unit or {})
                missing = [
                    name
                    for name in REQUIRED_FIELDS
                    if name not in raw or not _field_supplied(name, raw[name])
                ]
                raise AdmissionError(
                    f"{raw.get('id')!r}: missing required field(s) {missing}",
                    missing=missing,
                )
        except AdmissionError as exc:
            rec = {
                "id": (unit or fields).get("id") if isinstance(unit, dict) else fields.get("id"),
                "rejected": True,
                "reason": exc.reason,
                "missing": list(exc.missing),
            }
            self.rejected.append(rec)
            return {"kind": "rejected", "unit": None, "reason": exc.reason, "missing": list(exc.missing)}

        uid = built["id"]
        incoming_hash = graph_identity(built)
        existing = self.units.get(uid)
        if existing is not None:
            if graph_identity(existing) == incoming_hash:
                return {"kind": "idempotent", "unit": existing, "reason": None, "missing": []}
            raise IdentityConflict(
                f"WorkUnit id {uid!r} already exists with different content "
                f"(existing={graph_identity(existing)[:16]} incoming={incoming_hash[:16]})"
            )
        for other in self.units.values():
            if graph_identity(other) == incoming_hash:
                return {"kind": "idempotent", "unit": other, "reason": None, "missing": []}
        edges = list(built["dependencies"]) + list(built.get("verification_depends_on") or [])
        _assert_no_cycle(self.units, uid, edges)
        self.units[uid] = built
        return {"kind": "inserted", "unit": built, "reason": None, "missing": []}

    def compute_ready(self, *, mutate: bool = True) -> list[dict[str, Any]]:
        """Ready set is computed from edges. Never hand-ordered."""
        ready: list[dict[str, Any]] = []
        qualified = hardware_is_qualified()
        for uid in sorted(self.units):
            unit = self.units[uid]
            status = unit.get("status")
            if status in {"completed", "failed", "running"}:
                continue
            if status == "sleeping":
                if unit.get("requires_hardware") and not qualified:
                    continue
                if mutate:
                    unit["status"] = "pending"
                    if unit.get("classification") == "SLEEPING":
                        unit["classification"] = "STATIC_ONLY"
                    unit["blocked_reason"] = None
                else:
                    continue
            if not _deps_satisfied(unit, self.units):
                if mutate and unit.get("status") == "ready":
                    unit["status"] = "pending"
                continue
            if mutate and unit.get("status") in {"pending", "ready"}:
                unit["status"] = "ready"
            ready.append(unit)
        return ready

    def _occupied(self) -> dict[str, int]:
        counts: dict[str, int] = {lid: 0 for lid in LANE_IDS}
        for unit in self.units.values():
            if unit.get("status") == "running":
                lane = unit.get("resource_lane")
                if lane in counts:
                    counts[lane] += 1
        return counts

    def _physical_busy(self, scheduled: Sequence[Mapping[str, Any]]) -> set[str]:
        busy: set[str] = set()
        holders = list(scheduled) + [u for u in self.units.values() if u.get("status") == "running"]
        for unit in holders:
            spec = self.lanes.get(str(unit.get("resource_lane") or ""))
            if spec is not None and spec.physical:
                busy.add(spec.physical)
        return busy

    def _scope_holders(self, scheduled: Sequence[Mapping[str, Any]]) -> list[Sequence[str]]:
        holders = list(scheduled) + [u for u in self.units.values() if u.get("status") == "running"]
        return [list(u.get("mutation_scope") or []) for u in holders]

    def _can_place(
        self,
        unit: Mapping[str, Any],
        occupied: Mapping[str, int],
        scheduled: Sequence[Mapping[str, Any]],
    ) -> tuple[bool, str | None]:
        lane = str(unit["resource_lane"])
        spec = self.lanes[lane]
        used = int(occupied.get(lane, 0))
        if spec.exclusive and used >= 1:
            return False, "exclusive_lane"
        if used >= int(spec.capacity):
            return False, "capacity"
        if spec.physical:
            busy = self._physical_busy(scheduled)
            if spec.physical in busy:
                # Occupied by this lane or a sibling sharing the device.
                if used == 0:
                    return False, "physical_conflict"
                return False, "exclusive_lane"
        scope = list(unit.get("mutation_scope") or [])
        if scope:
            for other in self._scope_holders(scheduled):
                if scopes_intersect(scope, other):
                    return False, "mutation_scope"
        return True, None

    def tick(self) -> dict[str, Any]:
        """Claim a conflict-free ready set. Does not execute anyone."""
        self.tick_n += 1
        ready = self.compute_ready(mutate=True)
        occupied = self._occupied()
        scheduled: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        skip_reasons: dict[str, str] = {}
        ordered = select_order(ready)
        live_ids = [u["id"] for u in ordered]
        for unit in ordered:
            live = self.units[unit["id"]]
            ok, reason = self._can_place(live, occupied, scheduled)
            if not ok:
                live["skipped_ticks"] = int(live.get("skipped_ticks") or 0) + 1
                skipped.append(live)
                skip_reasons[live["id"]] = str(reason)
                continue
            live["status"] = "running"
            live["assigned_tick"] = self.tick_n
            live["skipped_ticks"] = 0
            occupied[live["resource_lane"]] = int(occupied.get(live["resource_lane"], 0)) + 1
            scheduled.append(live)

        reports: list[dict[str, Any]] = []
        for unit in skipped:
            skipped_n = int(unit.get("skipped_ticks") or 0)
            if skipped_n >= STARVATION_THRESHOLD:
                rec = {
                    "unit_id": unit["id"],
                    "resource_lane": unit["resource_lane"],
                    "skipped_ticks": skipped_n,
                    "reason": skip_reasons.get(unit["id"]),
                    "threshold": STARVATION_THRESHOLD,
                    "tick": self.tick_n,
                }
                reports.append(rec)
        # Lane-level: a lane had ready work every tick and none of it ran.
        ready_by_lane: dict[str, list[str]] = defaultdict(list)
        for unit in ordered:
            ready_by_lane[str(unit["resource_lane"])].append(unit["id"])
        scheduled_lanes = {u["resource_lane"] for u in scheduled}
        for lane, ids in sorted(ready_by_lane.items()):
            if lane in scheduled_lanes:
                continue
            if not ids:
                continue
            # If every ready unit on this lane was skipped for capacity/exclusive
            # held by a running (not this-tick) unit, that is occupancy wait.
            # Report once any of them crossed the threshold.
            if any(self.units[i].get("skipped_ticks", 0) >= STARVATION_THRESHOLD for i in ids):
                reports.append(
                    {
                        "unit_id": None,
                        "resource_lane": lane,
                        "skipped_ticks": max(int(self.units[i].get("skipped_ticks") or 0) for i in ids),
                        "reason": "lane_starvation",
                        "ready_ids": ids,
                        "threshold": STARVATION_THRESHOLD,
                        "tick": self.tick_n,
                    }
                )
        if reports:
            self.starvation_reports.extend(reports)

        schedule = {
            "tick": self.tick_n,
            "scheduled_ids": [u["id"] for u in scheduled],
            "scheduled_lanes": sorted({u["resource_lane"] for u in scheduled}),
            "ready_ids": live_ids,
            "skipped_ids": [u["id"] for u in skipped],
            "skip_reasons": skip_reasons,
            "running_ids": self.ids_in("running"),
            "completed_ids": self.ids_in("completed"),
            "sleeping_ids": self.ids_in("sleeping"),
            "starvation": reports,
            "executes": False,
            "evidence_class": "STATIC_ONLY",
            "gpu_authority": False,
        }
        self.last_schedule = schedule
        self.save()
        return schedule

    def record_result(self, unit_id: str, *, ok: bool) -> dict[str, Any]:
        """Executor-sibling callback. Never a path that invents a hardware result."""
        unit = self.units.get(unit_id)
        if unit is None:
            raise WorkGraphError(f"record_result: unknown unit {unit_id!r}")
        if unit.get("status") == "sleeping":
            raise SyntheticResultError(
                f"{unit_id}: SLEEPING hardware work cannot be completed with a synthetic result"
            )
        if unit.get("status") != "running":
            raise SyntheticResultError(
                f"{unit_id}: status {unit.get('status')!r} was never scheduled; "
                "refusing to complete unexecuted work"
            )
        if unit.get("requires_hardware") and not hardware_is_qualified():
            raise SyntheticResultError(
                f"{unit_id}: hardware still unqualified; refusing a fabricated completion"
            )
        unit["status"] = "completed" if ok else "failed"
        unit["completed_tick"] = self.tick_n
        unit["verification"] = {
            "ok": bool(ok),
            "evidence_class": "STATIC_ONLY",
            "bench_state": "UNKNOWN",
            "settles_physical_claim": False,
        }
        self.save()
        return dict(unit)

    def wake_sleeping(self, *, hardware_qualified: bool) -> list[str]:
        """Move SLEEPING -> pending only when told the hardware qualified.

        Passing True does not complete anyone and does not invent a measurement.
        """
        if not hardware_qualified:
            return []
        woken: list[str] = []
        for uid in sorted(self.units):
            unit = self.units[uid]
            if unit.get("status") != "sleeping":
                continue
            unit["status"] = "pending"
            if unit.get("classification") == "SLEEPING":
                unit["classification"] = "STATIC_ONLY"
            unit["blocked_reason"] = None
            woken.append(uid)
        self.save()
        return woken

    def execute(self, *_args: Any, **_kwargs: Any) -> None:
        raise ExecutionRefused(
            "WorkGraph.execute is refused: this module emits a schedule; "
            "execution is a sibling lane's concern"
        )

    def emit_hcli_workunit(self, unit_id: str) -> dict[str, Any]:
        """Project one graph unit onto the recovered HCLI WorkUnit field set."""
        unit = self.units.get(unit_id)
        if unit is None:
            raise WorkGraphError(f"emit_hcli_workunit: unknown unit {unit_id!r}")
        return graph_unit_to_hcli(unit)

    def emit_schedule_workunits(self) -> list[dict[str, Any]]:
        ids = list((self.last_schedule or {}).get("scheduled_ids") or self.ids_in("running"))
        return [graph_unit_to_hcli(self.units[i]) for i in ids if i in self.units]


def graph_unit_to_hcli(unit: Mapping[str, Any]) -> dict[str, Any]:
    status = str(unit.get("status") or "pending")
    hcli_status = {
        "pending": "pending",
        "ready": "ready",
        "running": "running",
        "completed": "completed",
        "failed": "failed",
        "sleeping": "blocked",
    }.get(status, "pending")
    classification = unit.get("classification")
    if status == "sleeping":
        classification = "SLEEPING"
    wu = WorkUnit(
        id=str(unit["id"]),
        role=str(unit.get("role") or "science"),
        description=str(unit.get("description") or ""),
        dependencies=list(unit.get("dependencies") or []),
        resource_class=str(unit.get("hcli_resource_class") or LANE_TO_HCLI[str(unit["resource_lane"])]),
        verifier=str(unit.get("verifier") or ""),
        effect_class=str(unit.get("effect_class") or "READ_ONLY"),
        workspace="repo-root",
        classification=classification,
        status="pending",
        provider="future.workgraph",
        repair_depth=0,
    )
    row = wu.to_dict()
    row.update(
        {
            "status": hcli_status,
            "classification": classification,
            "claim_boundary": unit.get("claim_boundary") or CLAIM_BOUNDARY,
            "requires_quiescence": str(unit.get("resource_lane")) == "GPU_PROTECTED",
            "species": unit.get("species"),
            "resource_lane": unit.get("resource_lane"),
            "mutation_scope": list(unit.get("mutation_scope") or []),
            "verification_depends_on": list(unit.get("verification_depends_on") or []),
            "expected_information_gain": unit.get("expected_information_gain"),
            "cost_units": unit.get("cost_units"),
            "requires_hardware": bool(unit.get("requires_hardware")),
            "blocked_reason": unit.get("blocked_reason"),
            "graph_status": status,
            "evidence_class": "STATIC_ONLY",
            "bench_state": "UNKNOWN",
            "gpu_authority": False,
            "may_promote": False,
            "may_modify_verifier": False,
        }
    )
    WorkUnit.from_dict(dict(row))
    return row


# ---------------------------------------------------------------------------
# Recovery ingest — HCLI future units + qualification queue as DATA
# ---------------------------------------------------------------------------


def _load_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        doc = load_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _candidates_by_id() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in (EVIDENCE_QUEUE,):
        doc = _load_json_if_present(path)
        if not doc:
            continue
        for row in doc.get("candidates") or []:
            if isinstance(row, dict) and row.get("candidate_id"):
                out[str(row["candidate_id"])] = row
    return out


def _stems_for(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    try:
        from tools.future.candidate_planner import SHARED_ENV_KEYS, distinctive_stems
    except Exception:
        env = {}
        raw = candidate.get("exact_mutation") or {}
        if isinstance(raw, Mapping):
            nested = raw.get("child_fusion_env") or raw.get("source_oracle_controls") or raw
            if isinstance(nested, Mapping):
                env = {str(k): str(v) for k, v in nested.items()}
        return tuple(sorted(k for k in env if k not in {"HAWKING_QWEN38_FAST"}))
    del SHARED_ENV_KEYS
    return tuple(sorted(distinctive_stems(candidate)))


def infer_lane(row: Mapping[str, Any]) -> str:
    species = str(row.get("species") or "")
    backend = str(row.get("preferred_backend") or row.get("backend") or "").lower()
    if species == "fpga_simulation" or backend == "fpga":
        return "FPGA_SIM"
    if species == "fusion_simulation":
        return "FPGA_SIM"
    if backend == "ane" or species and "ane" in species:
        return "ANE"
    if species == "learned_compiler_experiment":
        return "CPU_REPRESENTATION"
    if species == "independent_reproduction":
        return "CPU_VERIFY"
    if species in SPECIES_LANE:
        lane = SPECIES_LANE[species]
        if lane == "GPU_PROTECTED" and str(row.get("candidate_status") or "") == "READY_DIAGNOSTIC":
            return "GPU_DIAGNOSTIC"
        return lane
    rc = str(row.get("resource_class") or "")
    if rc in HCLI_TO_LANE:
        return HCLI_TO_LANE[rc]
    return "CPU_ANALYSIS"


def infer_info_cost(row: Mapping[str, Any], lane: str) -> tuple[int, int]:
    """Ranking proxies. Cite the hardware_doctor simulator ladder; not a measurement."""
    if lane == "GPU_PROTECTED":
        return INFO_HIGH, 6
    if lane in {"GPU_DIAGNOSTIC", "ANE"}:
        return INFO_MEDIUM, 4
    if lane == "FPGA_SIM":
        return INFO_MEDIUM, 3
    if lane == "CPU_VERIFY":
        return INFO_MEDIUM, 2
    if lane == "CPU_REPRESENTATION":
        return INFO_MEDIUM, 2
    if lane in {"CPU_ANALYSIS", "CPU_BUILD", "NETWORK_RESEARCH", "DISK_IO"}:
        return INFO_MEDIUM, 1
    return INFO_LOW, 1


def mutation_scope_for(row: Mapping[str, Any], candidates: Mapping[str, Mapping[str, Any]]) -> list[str]:
    scope: list[str] = []
    cid = row.get("candidate_id")
    if cid and str(cid) in candidates:
        cand = candidates[str(cid)]
        region = cand.get("affected_physical_region")
        if region:
            scope.append(f"region:{region}")
        for stem in _stems_for(cand):
            scope.append(f"stem:{stem}")
    receipt = row.get("output_receipt_path")
    if receipt:
        scope.append(f"receipt:{receipt}")
    return sorted(set(scope))


def recovered_hcli_units() -> tuple[list[dict[str, Any]], str]:
    """Load the species starting queue. Missing receipt is not absence of the queue."""
    doc = _load_json_if_present(WORKUNITS_RECEIPT)
    if doc and isinstance(doc.get("work_units"), list) and doc["work_units"]:
        return list(doc["work_units"]), str(WORKUNITS_RECEIPT.relative_to(REPO))
    try:
        from tools.future import workunit_species as ws

        return ws.build_starting_queue(), "tools.future.workunit_species.build_starting_queue"
    except Exception as exc:
        return [], f"unavailable:{type(exc).__name__}"


def ingest_recovered(graph: WorkGraph) -> dict[str, Any]:
    """Map recovered HCLI units onto graph units. Physical GPU/ANE -> SLEEPING."""
    rows, source = recovered_hcli_units()
    candidates = _candidates_by_id()
    inserted = 0
    sleeping = 0
    rejected = 0
    for row in rows:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        lane = infer_lane(row)
        info, cost = infer_info_cost(row, lane)
        hardware = lane in HARDWARE_LANES or bool(row.get("requires_quiescence"))
        try:
            outcome = graph.admit(
                make_unit(
                    id=str(row["id"]),
                    role=str(row.get("role") or "science"),
                    description=str(row.get("description") or row["id"]),
                    dependencies=list(row.get("dependencies") or []),
                    resource_lane=lane,
                    mutation_scope=mutation_scope_for(row, candidates),
                    verifier=str(row.get("verifier") or f"future.workgraph.{row['id']}"),
                    expected_information_gain=info,
                    cost_units=cost,
                    requires_hardware=hardware,
                    species=row.get("species"),
                    effect_class=str(row.get("effect_class") or "READ_ONLY"),
                    claim_boundary=str(row.get("claim_boundary") or CLAIM_BOUNDARY),
                    extras={
                        "candidate_id": row.get("candidate_id"),
                        "source_hcli_resource_class": row.get("resource_class"),
                        "source": source,
                    },
                )
            )
        except (AdmissionError, IdentityConflict, CycleError):
            rejected += 1
            continue
        if outcome["kind"] == "rejected":
            rejected += 1
            continue
        if outcome["kind"] == "inserted":
            inserted += 1
            if outcome["unit"] and outcome["unit"].get("status") == "sleeping":
                sleeping += 1
    return {
        "source": source,
        "source_rows": len(rows),
        "inserted": inserted,
        "sleeping": sleeping,
        "rejected": rejected,
        "graph_units": len(graph.units),
    }


# ---------------------------------------------------------------------------
# Negative controls — a guard nobody has watched fail is not a guard
# ---------------------------------------------------------------------------


def _probe_unit(uid: str, lane: str, **over: Any) -> dict[str, Any]:
    fields = dict(
        id=uid,
        role="science",
        description=f"schedule probe {uid} on {lane}",
        dependencies=[],
        resource_lane=lane,
        mutation_scope=[],
        verifier=f"future.workgraph.probe.{uid}",
        expected_information_gain=INFO_MEDIUM,
        cost_units=1,
        requires_hardware=False,
        species="workgraph_probe",
        effect_class="READ_ONLY",
    )
    fields.update(over)
    return make_unit(**fields)


def _prove_negative_controls(*, ncpu: int | None = None) -> dict[str, Any]:
    """Isolated proofs. Never touch the resident durable graph."""
    results: dict[str, Any] = {}
    ncpu = ncpu if ncpu is not None else max(8, _ncpu())

    g = WorkGraph(ncpu=ncpu)
    g.admit(_probe_unit("gpu-a", "GPU_PROTECTED", expected_information_gain=INFO_HIGH))
    g.admit(_probe_unit("gpu-b", "GPU_PROTECTED", expected_information_gain=INFO_LOW))
    g.admit(_probe_unit("cpu-1", "CPU_ANALYSIS"))
    g.admit(_probe_unit("research-1", "NETWORK_RESEARCH"))
    g.admit(_probe_unit("fpga-1", "FPGA_SIM"))
    s = g.tick()
    scheduled = set(s["scheduled_ids"])
    gpu_both = "gpu-a" in scheduled and "gpu-b" in scheduled
    concurrent = {"gpu-a", "cpu-1", "research-1"} <= scheduled or {"gpu-b", "cpu-1", "research-1"} <= scheduled
    fpga_too = "fpga-1" in scheduled
    results["gpu_protected_exclusive"] = {
        "fired": not gpu_both,
        "scheduled": sorted(scheduled),
        "lanes": list(s["scheduled_lanes"]),
        "note": "two GPU_PROTECTED units must never share a tick",
    }
    results["cpu_and_research_co_scheduled"] = {
        "fired": "cpu-1" in scheduled and "research-1" in scheduled,
        "scheduled": sorted(scheduled),
    }
    results["gpu_cpu_research_one_tick"] = {
        "fired": concurrent,
        "scheduled": sorted(scheduled),
        "fpga_also": fpga_too,
    }
    if gpu_both:
        raise WorkGraphError("negative control failed: two GPU_PROTECTED units co-scheduled")
    if not concurrent:
        raise WorkGraphError(
            f"negative control failed: GPU+CPU+research not in one tick: {sorted(scheduled)}"
        )

    g2 = WorkGraph(ncpu=ncpu)
    g2.admit(_probe_unit("mut-a", "CPU_ANALYSIS", mutation_scope=["crates/hawking-core/src/engine.rs"]))
    g2.admit(_probe_unit("mut-b", "CPU_ANALYSIS", mutation_scope=["crates/hawking-core/src/engine.rs", "docs/x"]))
    g2.admit(_probe_unit("mut-c", "CPU_ANALYSIS", mutation_scope=["receipts/future/WORKGRAPH_STATE.json"]))
    s2 = g2.tick()
    mut_ab = "mut-a" in s2["scheduled_ids"] and "mut-b" in s2["scheduled_ids"]
    mut_independent = "mut-c" in s2["scheduled_ids"]
    results["mutation_scope_blocks"] = {
        "fired": (not mut_ab) and mut_independent,
        "scheduled": list(s2["scheduled_ids"]),
        "skip_reasons": s2["skip_reasons"],
    }
    if mut_ab:
        raise WorkGraphError("negative control failed: intersecting mutation scopes co-scheduled")
    if not mut_independent:
        raise WorkGraphError("negative control failed: disjoint mutation scope was not scheduled")

    g3 = WorkGraph(ncpu=ncpu)
    outcome = g3.admit(
        {
            "id": "missing-verifier",
            "role": "science",
            "description": "should be rejected",
            "dependencies": [],
            "resource_lane": "CPU_ANALYSIS",
            "mutation_scope": [],
            "expected_information_gain": 1,
            "cost_units": 1,
        }
    )
    results["missing_field_rejected"] = {
        "fired": outcome["kind"] == "rejected",
        "reason": outcome.get("reason"),
        "missing": outcome.get("missing"),
        "not_in_graph": "missing-verifier" not in g3.units,
    }
    if outcome["kind"] != "rejected" or "missing-verifier" in g3.units:
        raise WorkGraphError("negative control failed: incomplete unit was admitted")

    g4 = WorkGraph(ncpu=ncpu)
    g4.admit(_probe_unit("parent", "CPU_ANALYSIS", expected_information_gain=INFO_HIGH))
    g4.admit(
        _probe_unit(
            "child",
            "CPU_VERIFY",
            verification_depends_on=["parent"],
            expected_information_gain=INFO_HIGH,
        )
    )
    s4 = g4.tick()
    child_waited = "child" not in s4["scheduled_ids"] and "parent" in s4["scheduled_ids"]
    g4.record_result("parent", ok=True)
    s4b = g4.tick()
    child_ran = "child" in s4b["scheduled_ids"]
    results["verification_dependency_waits"] = {
        "fired": child_waited and child_ran,
        "tick1": list(s4["scheduled_ids"]),
        "tick2": list(s4b["scheduled_ids"]),
    }
    if not (child_waited and child_ran):
        raise WorkGraphError("negative control failed: verification dependency did not wait")

    g5 = WorkGraph(ncpu=ncpu)
    g5.admit(_probe_unit("sleep-gpu", "GPU_PROTECTED", requires_hardware=True))
    assert g5.units["sleep-gpu"]["status"] == "sleeping"
    try:
        g5.record_result("sleep-gpu", ok=True)
        slept_guard = False
    except SyntheticResultError:
        slept_guard = True
    results["sleeping_not_synthetic"] = {
        "fired": slept_guard and g5.units["sleep-gpu"]["status"] == "sleeping",
        "status": g5.units["sleep-gpu"]["status"],
    }
    if not slept_guard:
        raise WorkGraphError("negative control failed: SLEEPING unit accepted a synthetic result")

    g6 = WorkGraph(ncpu=ncpu)
    try:
        g6.execute()
        exec_guard = False
    except ExecutionRefused:
        exec_guard = True
    results["execute_refused"] = {"fired": exec_guard}
    if not exec_guard:
        raise WorkGraphError("negative control failed: execute() did not refuse")

    g7b = WorkGraph(ncpu=ncpu)
    g7b.admit(_probe_unit("sel-low", "GPU_PROTECTED", expected_information_gain=INFO_LOW))
    g7b.admit(_probe_unit("sel-high", "GPU_PROTECTED", expected_information_gain=INFO_HIGH))
    s7 = g7b.tick()
    results["info_per_cost_wins"] = {
        "fired": s7["scheduled_ids"] == ["sel-high"],
        "scheduled": list(s7["scheduled_ids"]),
    }
    if s7["scheduled_ids"] != ["sel-high"]:
        raise WorkGraphError(f"negative control failed: selection was {s7['scheduled_ids']}")

    return results


def _prove_durability(workspace: Path) -> dict[str, Any]:
    g = WorkGraph(workspace=workspace)
    g.admit(_probe_unit("dur-a", "CPU_ANALYSIS", expected_information_gain=INFO_HIGH))
    g.admit(_probe_unit("dur-b", "NETWORK_RESEARCH"))
    g.admit(
        _probe_unit(
            "dur-c",
            "CPU_VERIFY",
            dependencies=["dur-a"],
            expected_information_gain=INFO_HIGH,
        )
    )
    s1 = g.tick()
    if "dur-a" not in s1["scheduled_ids"] or "dur-b" not in s1["scheduled_ids"]:
        raise WorkGraphError(f"durability setup tick failed: {s1['scheduled_ids']}")
    if "dur-c" in s1["scheduled_ids"]:
        raise WorkGraphError("dur-c must wait for dur-a")
    g.record_result("dur-a", ok=True)
    g.save()
    g2 = WorkGraph.load(workspace)
    if g2.units["dur-a"]["status"] != "completed":
        raise WorkGraphError("reload lost completed state; schedule would restart")
    if g2.units["dur-b"]["status"] != "running":
        raise WorkGraphError("reload lost running state")
    if g2.tick_n != g.tick_n:
        raise WorkGraphError("reload reset tick counter")
    s2 = g2.tick()
    if "dur-a" in s2["scheduled_ids"]:
        raise WorkGraphError("reload re-scheduled a completed unit (restart, not resume)")
    if "dur-c" not in s2["scheduled_ids"]:
        raise WorkGraphError("resume did not schedule the unblocked dependent")
    return {
        "fired": True,
        "tick1_scheduled": list(s1["scheduled_ids"]),
        "after_reload_completed": g2.units["dur-a"]["status"],
        "after_reload_running": g2.units["dur-b"]["status"],
        "tick2_scheduled": list(s2["scheduled_ids"]),
        "workspace": str(workspace),
    }


def _prove_starvation(*, ncpu: int | None = None) -> dict[str, Any]:
    ncpu = ncpu if ncpu is not None else max(8, _ncpu())
    g = WorkGraph(ncpu=ncpu)
    g.admit(_probe_unit("starve-low", "GPU_PROTECTED", expected_information_gain=INFO_LOW))
    reports: list[dict[str, Any]] = []
    scheduled_low_at: int | None = None
    for i in range(STARVATION_THRESHOLD + 2):
        high_id = f"starve-high-{i:02d}"
        g.admit(_probe_unit(high_id, "GPU_PROTECTED", expected_information_gain=INFO_HIGH))
        s = g.tick()
        if "starve-low" in s["scheduled_ids"]:
            scheduled_low_at = s["tick"]
            break
        reports.extend(s.get("starvation") or [])
        running = [uid for uid in s["scheduled_ids"] if uid != "starve-low"]
        if running:
            g.record_result(running[0], ok=True)
    if scheduled_low_at is None:
        raise WorkGraphError("starvation aging never scheduled the low-info unit")
    if not any(r.get("unit_id") == "starve-low" for r in g.starvation_reports):
        # Aging may schedule it on the threshold tick as it becomes boosted;
        # the report is written for skipped units the tick they are skipped.
        # Accept either a report or a boost-driven schedule at/after threshold.
        if scheduled_low_at < STARVATION_THRESHOLD:
            raise WorkGraphError("low-info unit ran before starvation aging could fire")
    return {
        "fired": True,
        "scheduled_low_at_tick": scheduled_low_at,
        "threshold": STARVATION_THRESHOLD,
        "reports": [r for r in g.starvation_reports if r.get("unit_id") == "starve-low"],
    }


# ---------------------------------------------------------------------------
# Resident-facing open / tick / receipt
# ---------------------------------------------------------------------------


def open_resident_graph(
    workspace: str | os.PathLike[str] | None = None,
    *,
    ncpu: int | None = None,
) -> WorkGraph:
    """Load the durable graph if it exists; otherwise ingest recovered units.

    Either checkout state is valid. Presence of a durable file means resume;
    absence means a fresh ingest. Tests must not treat absence as a code bug.
    """
    root = Path(workspace) if workspace is not None else DEFAULT_WORKSPACE
    path = root / DURABLE_NAME
    if path.is_file():
        return WorkGraph.load(root, ncpu=ncpu)
    g = WorkGraph(workspace=root, ncpu=ncpu)
    ingest_recovered(g)
    g.save()
    return g


def resident_callable() -> dict[str, Any]:
    return {
        "entry_point": (
            "python3 tools/future/workgraph.py --tick   "
            "# or: from tools.future.workgraph import open_resident_graph; "
            "open_resident_graph().tick()"
        ),
        "module": "tools.future.workgraph",
        "functions": {
            "open_resident_graph": "open_resident_graph(workspace=None) -> WorkGraph",
            "tick": "WorkGraph.tick() -> schedule dict (does not execute)",
            "admit": "WorkGraph.admit(unit|**fields) -> inserted|idempotent|rejected",
            "record_result": "WorkGraph.record_result(id, ok=bool)  # executor sibling",
            "wake_sleeping": "WorkGraph.wake_sleeping(hardware_qualified=bool)",
            "emit_hcli_workunit": "WorkGraph.emit_hcli_workunit(id) -> HCLI WorkUnit dict",
            "execute": "raises ExecutionRefused",
        },
        "workunit_emitted": {
            "constructor": "hcli.workunit.WorkUnit",
            "projection": "tools.future.workgraph.graph_unit_to_hcli",
            "resource_class": "mapped through LANE_TO_HCLI onto recovered HCLI ResourceClass",
            "status_sleeping": "HCLI status=blocked, classification=SLEEPING",
            "submit_target": (
                "hcli.scheduler.Scheduler.submit — integration point; this module "
                "does not construct a Scheduler (that would pull runtimes / Grok liveness)"
            ),
        },
        "receipt_written": f"receipts/future/{RECEIPT}",
        "durable_graph": str(DEFAULT_WORKSPACE / DURABLE_NAME),
        "frontier_fed": {
            "path": "receipts/future/CLAUDE_GLOBAL_FRONTIER.json",
            "related_entries": ["F002"],
            "mechanism": (
                "tick() emits the next ready set; completed units are the refill "
                "input. This module does not write CLAUDE_GLOBAL_FRONTIER.json "
                "(global_frontier.py is frozen for this lane). Disk authority is "
                f"receipts/future/{RECEIPT} plus the durable workgraph.json."
            ),
        },
        "fail_closed": [
            "admit() REJECTS a unit missing any required field (does not insert)",
            "unknown resource_lane is REJECTED",
            "dependency/verification cycle is REJECTED",
            "execute() raises ExecutionRefused",
            "record_result on SLEEPING raises SyntheticResultError",
            "record_result on never-scheduled raises SyntheticResultError",
            "record_result on requires_hardware while unqualified raises SyntheticResultError",
            "HardwareClaimError from write_receipt if a numeric hardware field appears",
            "durable document with the wrong schema raises DurableCorruptError",
            "GPU_PROTECTED physical work stays SLEEPING until hardware_qualified is true",
        ],
        "hcli_can_invoke": True,
        "note": (
            "HCLI Scheduler remains the execution dispatcher. WorkGraph is the "
            "graph planner the resident calls to decide WHAT is co-safe this tick. "
            "Sibling lanes (resident_api, sandbox, wakeup — not imported) consume "
            "the schedule."
        ),
    }


def recovered_implementation() -> dict[str, Any]:
    return {
        "hcli.scheduler.Scheduler": (
            "hcli/scheduler.py — admits WorkUnits, identify_ready + assign_ready, "
            "FIFO by ready_at (critical-path remaining_depth was retired; hops are "
            "not time and starved). GPU_EXCLUSIVE exclusive vs GPU_DECODE. "
            "MutationLock. DagStore durability. Does not invent work. complete() "
            "requires a passing verifier. from_workspace recovers running Grok."
        ),
        "hcli.workunit.WorkUnit": (
            "canonical unit; identify_ready computes the ready set from "
            "dependencies; assign_ready is occupancy, not a graph selector"
        ),
        "hcli.dag_store.DagStore": (
            "disk is authority (<workspace>/.hcli/dag.json); running-at-crash "
            "becomes interrupted (not a verifier failure). WorkGraph does not "
            "execute, so running is a claimed schedule slot and is restored as "
            "running on reload, not interrupted."
        ),
        "hcli.resources.ResourceClass": (
            "fourteen HCLI classes exist (GPU_DECODE, GPU_EXCLUSIVE, ...) but they "
            "are not the fourteen WorkGraph lanes. LANE_TO_HCLI is the projection."
        ),
        "hcli.agentos.runtime.AgentOS": (
            "composition facade; Mission/Scheduler remain authorities. "
            "WorkGraph is a planner the facade can call; it does not wrap AgentOS."
        ),
        "hcli.agentos.states.AgentState": (
            "READY/RUNNING/WAITING_RESOURCE/BLOCKED vocabulary; SLEEPING is a "
            "WorkGraph status projected to HCLI blocked+SLEEPING"
        ),
        "hcli.physical_graph.PhysicalGraph": (
            "placement/dataflow plan, PLAN_ONLY; not a work scheduler. Not forked."
        ),
        "tools.future.workunit_species": (
            "ten species, HCLI field set, starting queue. Ingested as DATA / via "
            "build_starting_queue; species authority constructor is not reimplemented."
        ),
        "tools.future.candidate_planner.conflict_reasons": (
            "same-region incompatible mutation and env-key collision. WorkGraph "
            "stores mutation_scope on each unit and refuses intersecting scopes "
            "at co-schedule time — the same refusal, on the live graph."
        ),
        "tools.future.hardware_doctor.rank_queue": (
            "integer key -(info*60//cost); reused, not rivalled"
        ),
        "tools.future.resident_optimizer.rank_hypotheses": (
            "same integer rule; reused for WorkGraph selection"
        ),
        "tools.future.qualification_pipeline": (
            "contamination_class HEAVY, lease_present fail-closed false; read as "
            "a receipt, never imported (importing would pull lease inspection)"
        ),
        "gap_that_existed": (
            "HCLI Scheduler is a list dispatcher on ResourceClass occupancy. It "
            "does not name the fourteen lanes, does not rank by information per "
            "cost, does not carry mutation_scope, does not report starvation, and "
            "does not SLEEP unqualified physical work as a first-class state. "
            "WorkGraph closes that gap around the recovered dispatcher rather "
            "than replacing it."
        ),
    }


def gaps_closed() -> list[str]:
    return [
        "graph, not list: nodes are WorkUnits, edges are dependency + verification; ready set is computed",
        "fourteen resource lanes with declared exclusivity and capacity; GPU_PROTECTED exclusive, CPU_ANALYSIS not",
        "exclusive-lane, shared-physical, and intersecting mutation_scope never co-schedule",
        "independent lanes (GPU probe, CPU_ANALYSIS, NETWORK_RESEARCH, FPGA_SIM) schedule in one tick",
        "admission REJECTS a unit missing any required field; the guard is watched to fail",
        "durable workgraph.json survives process restart; completed stays completed, running stays running, dependents resume",
        "selection is expected information gain per unit cost (hardware_doctor integer key) with deterministic ties",
        "starvation aging + reports so a low-info unit on an exclusive lane is not skipped indefinitely",
        "physical GPU/ANE recovered units SLEEP until hardware qualifies; record_result on them raises",
        "execute() does not exist as a capability — ExecutionRefused is the API",
        "HCLI WorkUnit projection so the resident can submit the schedule without this module dispatching",
    ]


def negative_findings() -> list[str]:
    return [
        "HCLI Scheduler.dispatch is FIFO, not information-per-cost; WorkGraph does not monkey-patch it",
        "HCLI ResourceClass names are not the fourteen lanes; a mapping layer is required and is an integration point",
        "this sidecar has no GPU authority; every GPU_PROTECTED recovered unit is SLEEPING, not scheduled, not completed",
        "protected bench lock holder pids are unproven; this module does not flock (flock would be a seizure)",
        "qualification pipeline classifies the machine HEAVY; WorkGraph does not quiesce anyone",
        "Flash NX remains SCAFFOLD_ONLY; Flash candidates stay SLEEPING with that reason on disk",
        "teacher capture 0/256 is recorded as a blocker, not turned into a synthetic corpus",
        "cannot import this-wave siblings (wakeup, resident_api, sandbox, super_resident, frontiers, …); local interfaces defined here",
        "cannot write hcli/**; the resident must call WorkGraph and then Scheduler.submit itself",
        "cannot write CLAUDE_GLOBAL_FRONTIER.json; frontier feed is by receipt, not by mutating that module",
        "DagStore interrupt-on-crash semantics are for execution; WorkGraph running means claimed, so reload keeps running",
        "NETWORK_RESEARCH is a lane declaration; this process never opens a socket",
        "GROK_SWARM capacity is the HCLI grok fallback (2), not a live GrokBridge census",
        "no Era VI and no Odyssey IV; FPGA is not its own civilization",
    ]


def next_workunits() -> list[dict[str, Any]]:
    return [
        {
            "id": "future.workgraph.wire-scheduler-submit",
            "resource_lane": "CPU_ANALYSIS",
            "description": (
                "Sibling resident_api / super_resident submits emit_hcli_workunit() "
                "rows into hcli.scheduler.Scheduler.submit. Not done here: hcli is frozen."
            ),
            "blocked_on": "resident_api.py (this-wave sibling, not imported)",
        },
        {
            "id": "future.workgraph.wakeup-sleeping-on-qualify",
            "resource_lane": "CPU_ANALYSIS",
            "description": (
                "Sibling wakeup.py calls WorkGraph.wake_sleeping(hardware_qualified=True) "
                "when Codex's qualification queue reports a real GPU window. Until then "
                "SLEEPING stays SLEEPING."
            ),
            "blocked_on": "wakeup.py (this-wave sibling) + Codex hardware qualification",
        },
        {
            "id": "future.workgraph.executor-record-result",
            "resource_lane": "CPU_VERIFY",
            "description": (
                "Sibling sandbox/executor calls record_result(id, ok=...) after a real "
                "verifier outcome. WorkGraph will not complete unexecuted work."
            ),
            "blocked_on": "sandbox.py (this-wave sibling)",
        },
    ]


def build(*, workspace: Path | None = None, ncpu: int | None = None) -> Path:
    proofs = _prove_negative_controls(ncpu=ncpu)
    starve = _prove_starvation(ncpu=ncpu)
    import tempfile

    dur_dir = Path(tempfile.mkdtemp(prefix="workgraph-durability-"))
    try:
        durability = _prove_durability(dur_dir)
    finally:
        # Leave no scratch. The proof already ran.
        try:
            for child in dur_dir.rglob("*"):
                if child.is_file():
                    child.unlink()
            dur_dir.rmdir()
        except OSError:
            pass

    root = workspace if workspace is not None else DEFAULT_WORKSPACE
    # Fresh recovered graph for the receipt snapshot so a later --selftest resume
    # cannot hide the first concurrent claim behind already-running occupancy.
    snap_ncpu = ncpu if ncpu is not None else max(8, _ncpu())
    snap = WorkGraph(ncpu=snap_ncpu)
    ingest_stats = ingest_recovered(snap)
    first_tick = snap.tick()
    snap_running = snap.ids_in("running")
    snap_lanes = sorted({snap.units[i]["resource_lane"] for i in snap_running})
    ingest_meta = {
        "resumed": False,
        "source": ingest_stats.get("source"),
        "source_rows": ingest_stats.get("source_rows"),
        "inserted": ingest_stats.get("inserted"),
        "units": len(snap.units),
        "by_status": _count_by_status(snap),
        "by_lane": _count_by_lane(snap),
    }
    # Resident durable graph: resume if present, ingest if not.
    graph = open_resident_graph(root, ncpu=ncpu)
    graph.tick()
    running_ids = graph.ids_in("running")
    running_lanes = sorted({graph.units[i]["resource_lane"] for i in running_ids})
    emitted = [graph_unit_to_hcli(snap.units[i]) for i in snap_running]

    qual = hardware_qualification()
    lanes_doc = [graph.lanes[lid].to_dict() for lid in LANE_IDS]
    frontier_present = FRONTIER_RECEIPT.is_file()

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Graph scheduler for the HCLI super-resident: fourteen resource lanes, "
            "computed ready set, mutation and exclusive-lane conflicts, information-"
            "per-cost selection, durable resume, SLEEPING unqualified hardware. "
            "Emits a schedule. Does not execute."
        ),
        "eras": list(ERAS),
        "odysseys": list(ODYSSEYS),
        "no_era_vi": True,
        "no_odyssey_iv": True,
        "fpga": (
            "FPGA is part of Accelerator / Physical Compiler / Fusion. "
            "FPGA_SIM is a CPU lane. This module does not build an FPGA backend."
        ),
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "lanes": lanes_doc,
        "lane_ids": list(LANE_IDS),
        "lane_to_hcli": dict(LANE_TO_HCLI),
        "selection_rule": {
            "key": "(-starvation_boost, -(info * 60 // cost), cost, id)",
            "info_domain": [INFO_LOW, INFO_MEDIUM, INFO_HIGH],
            "starvation_threshold_ticks": STARVATION_THRESHOLD,
            "source": (
                "tools/future/hardware_doctor.py::rank_queue and "
                "tools/future/resident_optimizer.py::rank_hypotheses"
            ),
            "never_a_hardware_measurement": True,
        },
        "hardware_qualification": qual,
        "recovered_graph": ingest_meta,
        "recovered_tick": {
            "tick": first_tick["tick"],
            "scheduled_ids": first_tick["scheduled_ids"],
            "scheduled_lanes": first_tick["scheduled_lanes"],
            "running_ids": snap_running,
            "running_lanes": snap_lanes,
            "sleeping": len(first_tick["sleeping_ids"]),
            "ready": len(first_tick["ready_ids"]),
            "note": (
                "Fresh recovered-graph tick (not the resumed durable occupancy). "
                "Independent non-GPU lanes claim together; GPU_PROTECTED recovered "
                "work stays SLEEPING."
            ),
            "executes": False,
        },
        "current_schedule": {
            "running_ids": running_ids,
            "running_lanes": running_lanes,
            "sleeping_count": len(graph.ids_in("sleeping")),
            "concurrent_non_gpu": sorted(
                lid for lid in running_lanes if lid not in HARDWARE_LANES
            ),
            "durable_resumed": graph.resumed,
        },
        "emitted_hcli_workunits": emitted,
        "concurrent_tick_proof": proofs.get("gpu_cpu_research_one_tick"),
        "negative_controls": proofs,
        "durability_proof": {k: v for k, v in durability.items() if k != "workspace"},
        "starvation_proof": starve,
        "graph": {
            "node_count": len(graph.units),
            "edge_count": _edge_count(graph),
            "edges": _edges(graph)[:80],
            "ready_ids": first_tick["ready_ids"],
            "running_ids": first_tick["running_ids"],
            "completed_ids": first_tick["completed_ids"],
            "sleeping_ids": first_tick["sleeping_ids"][:40],
            "note": (
                "ready set is computed; the lists here are a snapshot. "
                "edge_count is derived from units, not a fixed bound. "
                "sleeping_ids may be truncated in the receipt; the durable "
                "document holds the full graph."
            ),
        },
        "durable_path": str((graph.path or (DEFAULT_WORKSPACE / DURABLE_NAME)).relative_to(REPO))
        if graph.path and _under_repo(graph.path)
        else str(DEFAULT_WORKSPACE / DURABLE_NAME),
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(),
        "resident_callable": resident_callable(),
        "frontier_fed": resident_callable()["frontier_fed"],
        "frontier_receipt_present": frontier_present,
        "integration_points": [
            "hcli.scheduler.Scheduler.submit — consume emit_hcli_workunit() rows (hcli frozen)",
            "tools.future.wakeup — wake_sleeping(hardware_qualified=True) when Codex qualifies (sibling, not imported)",
            "tools.future.sandbox / resident_api — record_result after a real verifier (siblings, not imported)",
            "tools.future.frontiers / super_resident — refill CLAUDE_GLOBAL_FRONTIER from completed units (siblings, not imported)",
            "tools.future.candidate_planner.conflict_reasons — mutation_scope is the graph form of those conflict edges",
        ],
        "next_workunits": next_workunits(),
        "required_fields": list(REQUIRED_FIELDS),
        "statuses": sorted(GRAPH_STATUSES),
        "claim_boundary": CLAIM_BOUNDARY,
        "executes": False,
        "gpu_authority": False,
        "evidence_class": "STATIC_ONLY",
        "measurement_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    proofs = _prove_negative_controls()
    for name, row in proofs.items():
        if isinstance(row, dict) and row.get("fired") is False:
            raise AssertionError(f"selftest negative control {name} did not fire")
    _prove_starvation()
    return build()


def _count_by_status(graph: WorkGraph) -> dict[str, int]:
    counts: dict[str, int] = {}
    for unit in graph.units.values():
        key = str(unit.get("status") or "null")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _count_by_lane(graph: WorkGraph) -> dict[str, int]:
    counts: dict[str, int] = {}
    for unit in graph.units.values():
        key = str(unit.get("resource_lane") or "null")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _edges(graph: WorkGraph) -> list[dict[str, str]]:
    edges: list[dict[str, str]] = []
    for uid in sorted(graph.units):
        unit = graph.units[uid]
        for dep in unit.get("dependencies") or []:
            edges.append({"from": str(dep), "to": uid, "kind": "dependency"})
        for dep in unit.get("verification_depends_on") or []:
            edges.append({"from": str(dep), "to": uid, "kind": "verification"})
    edges.sort(key=lambda e: (e["from"], e["to"], e["kind"]))
    return edges


def _edge_count(graph: WorkGraph) -> int:
    return len(_edges(graph))


def _under_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(REPO.resolve())
        return True
    except ValueError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tick", action="store_true", help="open the resident graph and emit one schedule tick")
    ap.add_argument("--workspace", type=str, default=None)
    args = ap.parse_args()
    if args.selftest:
        print(selftest())
        return 0
    if args.tick:
        g = open_resident_graph(args.workspace)
        schedule = g.tick()
        print(json.dumps(schedule, indent=2, sort_keys=True))
        build(workspace=g.workspace)
        return 0
    print(build(workspace=Path(args.workspace) if args.workspace else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
