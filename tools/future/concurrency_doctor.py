"""RESIDENT CONCURRENCY DOCTOR — settle whether a second stream earns more useful work.

The machine may show partial GPU occupancy while the resident waits on a
subprocess. That occupancy is a HYPOTHESIS about usable headroom, not
available compute, and a utilisation percentage is not a complete-token.

This doctor plans the session-concurrency ladder (1, 2, 3, 4, stopping when
uninformative), consumes per-level observations, and verdicts on VERIFIED
USEFUL WORK PER WALL SECOND. A configuration at 95% GPU occupancy that
produces fewer useful experiments per hour LOSES.

This sidecar has no GPU lease and no protected bench. It records the PLAN
and the host's current capability to run it. It does not take a lease, does
not run a protected benchmark, and does not record a hardware performance
number. Absent a resident process — or absent a lease — the experiment is
SLEEPING with a wake condition.

A resulting law is SCOPED to this machine, this NX, this runtime, and this
context regime. It is not a Flash, M5, FPGA, or CUDA law. The recovered
4 MiB kernel sweep and the whole-body bandwidth law live in different
regimes; mixing them is the failure this scoping exists to prevent.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
from typing import Any, Mapping, Sequence

from tools.future import contamination as C
from tools.future import hardware_doctor as hd
from tools.future import resident_health as rh
from tools.future import workunit_species as wus
from tools.future._common import (
    HARDWARE_FIELDS,
    HardwareClaimError,
    RECEIPTS,
    REPO,
    _assert_no_hardware_claims,
    write_receipt,
)

RECEIPT = "RESIDENT_CONCURRENCY_DOCTOR.json"
SCHEMA = "hawking.future.concurrency_doctor.v1"
RECORDED_BY = "tools/future/concurrency_doctor.py"
VERSION = 1

LEVELS: tuple[int, ...] = (1, 2, 3, 4)
VERDICTS: tuple[str, ...] = (
    "CONCURRENCY_HELPS",
    "NO_USEFUL_CONCURRENCY_HEADROOM",
    "HEADROOM_IS_HOST_CEREMONY",
)
OCCUPANCY_CLASSES: tuple[str, ...] = ("LOW", "MEDIUM", "HIGH", "UNKNOWN")
CEREMONY_CLASSES: tuple[str, ...] = ("LOW", "MEDIUM", "HIGH", "UNKNOWN")
DELTA_CLASSES: tuple[str, ...] = ("UP", "DOWN", "FLAT")
PRESENCE_PRESENT = "PRESENT"

SCOPE_FIELDS: tuple[str, ...] = ("machine", "nx", "runtime", "context_regime")
CONTEXT_REGIMES: tuple[str, ...] = (
    "WHOLE_MODEL_BODY",
    "SMALL_KERNEL_LAUNCH_BOUND",
    "HOST_CEREMONY",
    "BANDWIDTH_BOUND",
)
UNIVERSAL_REFUSALS: tuple[str, ...] = ("Flash", "M5", "FPGA", "CUDA")
GENERIC_MACHINES = frozenset(
    {
        "apple silicon",
        "any apple",
        "all apple silicon",
        "m5",
        "cuda",
        "fpga",
        "all macs",
        "any gpu",
        "any machine",
    }
)

# Class boundary, not a measured result. Below contamination's HEAVY_GPU_UTIL_PCT
# the device is quiet; at/above this the occupancy class is HIGH.
OCCUPANCY_HIGH_MIN_PCT = 80

# Informative iff useful work grew by more than one tenth. Integer compare so
# a 10% exact tie is FLAT, not a fabricated UP.
INFORMATIVE_NUM = 11
INFORMATIVE_DEN = 10

FRONTIER = "FT.TPS.protected-tps"
SPECIES = "CONCURRENCY_DOCTOR"
WORKUNIT_ID = "WU.CONCURRENCY_DOCTOR.concurrency_doctor"

OBSERVATION_SLOTS: tuple[str, ...] = (
    "per_session_useful_work",
    "aggregate_useful_work",
    "per_session_token_cost_class",
    "aggregate_token_cost_class",
    "ttft_class",
    "gpu_occupancy_class",
    "cpu_occupancy_class",
    "resident_bytes",
    "session_state_bytes",
    "memory_pressure",
    "swap",
    "dispatch_rate_class",
    "synchronisation_class",
    "host_ceremony_class",
)

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement. Does not produce "
    "DIAGNOSTIC_RELATIVE or PROTECTED_ABSOLUTE. Partial GPU occupancy is not "
    "available compute. A utilisation percentage is not a complete-token. "
    "SLEEPING physical work is never filled with a synthetic TPS."
)

WAKE_ALL_OF: tuple[str, ...] = (
    "a declared resident pid is PRESENT (tools.future.resident_health.sample); "
    "the largest RSS neighbour is not identity",
    "a protected GPU lease is held; this sidecar never takes one",
    "contamination class is QUIESCENT",
    "machine, NX, runtime, and context_regime are named so a law can be scoped",
)
WAKE_NEVER: tuple[str, ...] = (
    "synthetic useful-work numbers treated as PROTECTED_ABSOLUTE",
    "utilisation treated as available compute",
    "universalising the law to Flash, M5, FPGA, or CUDA",
    "a default of CONCURRENCY_HELPS when the ladder has not run",
)

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


class DoctorRefuse(ValueError):
    """Fail closed: missing input, no lease, or a guess that would look like success."""


class ObservationRefuse(DoctorRefuse):
    """observe() refused rather than invent a level."""


class VerdictRefuse(DoctorRefuse):
    """verdict() refused rather than default to CONCURRENCY_HELPS."""


class ScopeRefused(DoctorRefuse):
    """A law without machine/NX/runtime/context_regime, or one that universalises."""


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, str) and value.strip().upper() == "UNKNOWN":
        return False
    if isinstance(value, (dict, list, tuple)) and not value:
        return False
    return True


def refuse_hardware_named_number(name: str, value: Any) -> None:
    """A hardware-named numeric field is a measurement claim. Raise, do not store."""
    base = name.rsplit(".", 1)[-1]
    if base in HARDWARE_FIELDS and isinstance(value, (int, float)) and not isinstance(value, bool):
        raise HardwareClaimError(
            f"{name}={value!r}: sidecar has no GPU authority; hardware-named "
            "fields must be null/UNKNOWN"
        )


def refuse_hardware_tree(node: Any, path: str = "") -> None:
    _assert_no_hardware_claims(node)
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else key
            refuse_hardware_named_number(here, value)
            refuse_hardware_tree(value, here)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            refuse_hardware_tree(value, f"{path}[{i}]")


def occupancy_class_from_pct(pct: Any) -> str:
    """Occupancy class is machine-state taxonomy, not available compute."""
    if not isinstance(pct, (int, float)) or isinstance(pct, bool):
        return "UNKNOWN"
    if pct < 0:
        return "UNKNOWN"
    if pct < C.HEAVY_GPU_UTIL_PCT:
        return "LOW"
    if pct >= OCCUPANCY_HIGH_MIN_PCT:
        return "HIGH"
    return "MEDIUM"


def unknown_slots() -> dict[str, Any]:
    slots = {name: None for name in OBSERVATION_SLOTS}
    for name in HARDWARE_FIELDS:
        slots[name] = None
    return slots


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


def plan() -> dict[str, Any]:
    """The experiment. Continue only while informative. Occupancy is never why."""
    levels = (
        {
            "concurrency": 1,
            "tells_us": (
                "control: one stream's useful work per wall second, and the "
                "split between GPU occupancy and host ceremony (CPU prep / "
                "readback / sync). Always run."
            ),
            "continue_if": "always; this is the control",
        },
        {
            "concurrency": 2,
            "tells_us": (
                "whether a second stream increases useful work per wall second "
                "or only occupancy and contention. First falsifier of "
                "'partial occupancy is free headroom'."
            ),
            "continue_if": (
                "useful work at 2 is UP vs 1; occupancy rising is not informative"
            ),
        },
        {
            "concurrency": 3,
            "tells_us": (
                "whether the gain continues or the knee is at 2. Only run if 2 "
                "was informative."
            ),
            "continue_if": "useful work at 3 is UP vs 2",
        },
        {
            "concurrency": 4,
            "tells_us": (
                "last probe this doctor will ask for. Diminishing returns vs 3. "
                "Stop after 4 regardless."
            ),
            "continue_if": "useful work at 4 is UP vs 3; ladder ends here either way",
        },
    )
    return {
        "objective": "verified useful work per wall second, never utilisation",
        "levels": list(levels),
        "ladder": list(LEVELS),
        "stop_rule": (
            "continue only while useful work is UP vs the previous level; "
            "an uninformative step ends the ladder. Occupancy change is never "
            "a reason to continue."
        ),
        "informative_if": (
            f"useful_work_per_wall_second(n+1) * {INFORMATIVE_DEN} > "
            f"useful_work_per_wall_second(n) * {INFORMATIVE_NUM} "
            "(more than one tenth). Exact-tenth is FLAT."
        ),
        "required_observation_slots": list(OBSERVATION_SLOTS),
        "hardware_named_slots_must_be_null": sorted(HARDWARE_FIELDS),
        "required_scope_to_persist": list(SCOPE_FIELDS),
        "legal_verdicts": list(VERDICTS),
        "context_regimes": list(CONTEXT_REGIMES),
        "does_not_universalise_to": list(UNIVERSAL_REFUSALS),
        "resource_class_when_awake": "GPU_EXCLUSIVE",
        "resource_class_while_planning": "CPU_ANALYSIS",
        "evidence_class_when_run": (
            "PROTECTED_ABSOLUTE on a lease this sidecar does not hold"
        ),
        "evidence_class_here": "STATIC_ONLY",
        "what_it_refuses": [
            "a default of CONCURRENCY_HELPS",
            "utilisation as the ranking key",
            "taking a GPU lease",
            "recording tps / token_ns / gpu_ns / bandwidth_gbps",
            "a law without machine, NX, runtime, and context_regime",
            "transfer to Flash, M5, FPGA, or CUDA",
        ],
        "wake_all_of": list(WAKE_ALL_OF),
        "wake_never": list(WAKE_NEVER),
        "frontier": FRONTIER,
        "related_frontiers": (
            "FT.TPS.accepted-token-cost",
            "FT.LATENCY.gpu-ns",
            "FT.LATENCY.cpu-turnaround",
        ),
    }


# ---------------------------------------------------------------------------
# capability / sleeping
# ---------------------------------------------------------------------------


def _not_runnable_reasons(cap: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    presence = cap.get("resident_presence")
    if presence != PRESENCE_PRESENT:
        reasons.append(
            f"resident presence is {presence!r}, not PRESENT; will not invent a "
            "pid from the largest RSS neighbour"
        )
    if cap.get("gpu_authority") is not True:
        reasons.append("sidecar has no GPU authority and will not take a lease")
    if cap.get("protected_lease") is not True:
        reasons.append("no protected GPU lease is held")
    if cap.get("quiescence") != "QUIESCENT":
        reasons.append(
            f"quiescence is {cap.get('quiescence')!r}, not QUIESCENT; a "
            "DIAGNOSTIC_RELATIVE number on a busy machine is not this experiment"
        )
    reasons.append(
        "this sidecar is structurally unable to execute the protected ladder; "
        "a process with PROTECTED_ABSOLUTE authority must run observe() on hardware"
    )
    return reasons


def host_capability(
    *,
    resident_sample: Mapping[str, Any] | None = None,
    metal: Mapping[str, Any] | None = None,
    occupancy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """What this host can do today. Capability, never a concurrency measurement."""
    metal_doc = dict(metal) if metal is not None else hd.metal_state()
    sample = dict(resident_sample) if resident_sample is not None else rh.sample()
    resident = sample.get("resident") if isinstance(sample.get("resident"), dict) else {}
    presence = resident.get("presence") or "UNDECLARED"
    mem = sample.get("memory") if isinstance(sample.get("memory"), dict) else {}

    if occupancy is None:
        occupancy = C.probe_gpu_occupancy()
    occ = dict(occupancy)
    occ_class = (
        occupancy_class_from_pct(occ.get("device_utilization_pct"))
        if occ.get("status") == "OK"
        else "UNKNOWN"
    )

    chip = metal_doc.get("chip")
    cap: dict[str, Any] = {
        "resident_presence": presence,
        "resident_pid": resident.get("pid"),
        "resident_bytes": resident.get("rss_bytes"),
        "resident_reason": resident.get("reason"),
        "gpu_authority": False,
        "protected_lease": False,
        "quiescence": "UNKNOWN",
        "quiescence_reason": (
            "this sidecar did not run contamination.snapshot as a promotion "
            "gate; UNKNOWN is not QUIESCENT"
        ),
        "metal": {
            "chip": chip,
            "gpu_present": bool(metal_doc.get("gpu_present")),
            "offline_metal_compiler": bool(metal_doc.get("offline_metal_compiler")),
            "runtime_source_compilation": metal_doc.get("runtime_source_compilation"),
            "is_a_measurement": False,
            "why": metal_doc.get("why_gpu") or metal_doc.get("why_runtime_state"),
        },
        "gpu_occupancy_class": occ_class,
        "gpu_occupancy_probe_status": occ.get("status"),
        "occupancy_is_not_available_compute": True,
        "occupancy_is_hypothesis_about_headroom": True,
        "memory_pressure": mem.get("uma_pressure_name") or "UNKNOWN",
        "swap": {
            "status": mem.get("status") or "UNKNOWN",
            "swap_ins": mem.get("swap_ins") if isinstance(mem.get("swap_ins"), int) else None,
            "note": "cumulative vm_stat Swapins, not a per-level experiment rate",
        },
        "runnable_today": False,
        "is_a_measurement": False,
        "evidence_class": "STATIC_ONLY",
        "suggested_scope": {
            "machine": chip if _present(chip) and str(chip).lower() != "unknown" else None,
            "nx": None,
            "runtime": None,
            "context_regime": None,
            "note": (
                "hints only; seal_law refuses UNKNOWN/missing and will not auto-fill"
            ),
        },
    }
    cap["not_runnable_reasons"] = _not_runnable_reasons(cap)
    refuse_hardware_tree(cap)
    return cap


def emit_sleeping_workunit(capability: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """SLEEPING unit with a wake condition. Never a synthetic COMPLETED."""
    cap = dict(capability) if capability is not None else {}
    reasons = list(cap.get("not_runnable_reasons") or _not_runnable_reasons(cap))
    extras: dict[str, Any] = {
        "frontier": FRONTIER,
        "species": SPECIES,
        "wakeup_state": "SLEEPING",
        "sleeping": True,
        "candidate_status": "SLEEPING",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "bench_state": "UNKNOWN",
        "wake_condition": {
            "all_of": list(WAKE_ALL_OF),
            "never": list(WAKE_NEVER),
        },
        "blocked_reason": "; ".join(reasons) if reasons else "experiment is not runnable on this sidecar",
        "required_lanes": ["GPU_PROTECTED"],
        "claim_boundary": CLAIM_BOUNDARY,
        "output_receipt_path": f"receipts/future/{RECEIPT}",
        "requires_quiescence": True,
    }
    row = wus.emit_hcli_workunit(
        id=WORKUNIT_ID,
        role="science",
        description=(
            "SLEEPING resident-session concurrency ladder (1, 2, 3, 4 while "
            "informative). Wakes when a declared resident is PRESENT, a "
            "protected GPU lease is held, and quiescence is QUIESCENT. Never a "
            "synthetic TPS. Objective is useful work per wall second, never "
            "utilisation."
        ),
        dependencies=[],
        resource_class="GPU_EXCLUSIVE",
        verifier="future.concurrency_doctor.verdict",
        provider="future.concurrency_doctor",
        effect_class="READ_ONLY",
        status="blocked",
        classification="SLEEPING",
        extras=extras,
    )
    wus.validate_emitted_unit(row)
    if row.get("status") != "blocked" or row.get("classification") != "SLEEPING":
        raise DoctorRefuse("sleeping unit lost its SLEEPING classification")
    if row.get("verdict") == "CONCURRENCY_HELPS":
        raise DoctorRefuse("sleeping unit must not carry a CONCURRENCY_HELPS verdict")
    return row


def decide(capability: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Always SLEEPING on this sidecar. Never a default CONCURRENCY_HELPS."""
    cap = dict(capability) if capability is not None else host_capability()
    reasons = list(cap.get("not_runnable_reasons") or _not_runnable_reasons(cap))
    unit = emit_sleeping_workunit(cap)
    decision = {
        "experiment_state": "SLEEPING",
        "verdict": None,
        "reason": "; ".join(reasons),
        "resident_presence": cap.get("resident_presence"),
        "workunit": unit,
        "wake_condition": unit.get("wake_condition"),
        "gpu_authority": False,
        "evidence_class": "STATIC_ONLY",
        "is_a_measurement": False,
    }
    if decision["verdict"] is not None:
        raise DoctorRefuse("decide() invented a verdict; this sidecar cannot")
    return decision


# ---------------------------------------------------------------------------
# observe
# ---------------------------------------------------------------------------


def make_synthetic_observation(*, concurrency: int, **fields: Any) -> dict[str, Any]:
    """In-memory observation for verdict reachability. Not a live measurement.

    useful_work_per_wall_second is a synthetic ranking key, not a TPS field.
    Hardware-named numbers raise.
    """
    for key, value in fields.items():
        refuse_hardware_named_number(key, value)
    if concurrency not in LEVELS:
        raise ObservationRefuse(f"concurrency {concurrency} is not on the ladder {LEVELS}")
    work = fields.get("useful_work_per_wall_second")
    if not isinstance(work, (int, float)) or isinstance(work, bool) or work <= 0:
        raise ObservationRefuse(
            f"concurrency {concurrency}: useful_work_per_wall_second is {work!r}; "
            "missing/zero is not a result"
        )
    gpu = fields.get("gpu_occupancy_class", "UNKNOWN")
    ceremony = fields.get("host_ceremony_class", "UNKNOWN")
    if gpu not in OCCUPANCY_CLASSES:
        raise ObservationRefuse(f"gpu_occupancy_class {gpu!r} is not in {OCCUPANCY_CLASSES}")
    if ceremony not in CEREMONY_CLASSES:
        raise ObservationRefuse(
            f"host_ceremony_class {ceremony!r} is not in {CEREMONY_CLASSES}"
        )
    cpu = fields.get("cpu_occupancy_class", "UNKNOWN")
    if cpu not in OCCUPANCY_CLASSES:
        raise ObservationRefuse(f"cpu_occupancy_class {cpu!r} is not in {OCCUPANCY_CLASSES}")
    per_session = fields.get("per_session_useful_work")
    if per_session is not None:
        if not isinstance(per_session, (list, tuple)) or len(per_session) != concurrency:
            raise ObservationRefuse(
                f"per_session_useful_work length {len(per_session) if isinstance(per_session, (list, tuple)) else None} "
                f"!= concurrency {concurrency}"
            )
        for item in per_session:
            if not isinstance(item, (int, float)) or isinstance(item, bool) or item <= 0:
                raise ObservationRefuse(f"per-session useful work {item!r} is not a positive quantity")
    obs = {
        "status": "OK",
        "source": "SYNTHETIC",
        "concurrency": concurrency,
        "useful_work_per_wall_second": float(work),
        "per_session_useful_work": list(per_session) if per_session is not None else None,
        "gpu_occupancy_class": gpu,
        "cpu_occupancy_class": cpu,
        "host_ceremony_class": ceremony,
        "ttft_class": fields.get("ttft_class") or "UNKNOWN",
        "dispatch_rate_class": fields.get("dispatch_rate_class") or "UNKNOWN",
        "synchronisation_class": fields.get("synchronisation_class") or "UNKNOWN",
        "resident_bytes": fields.get("resident_bytes"),
        "session_state_bytes": fields.get("session_state_bytes"),
        "memory_pressure": fields.get("memory_pressure") or "UNKNOWN",
        "swap": fields.get("swap"),
        "reason": fields.get("reason") or "synthetic observation for verdict reachability",
        "gpu_authority": False,
        "evidence_class": "SYNTHETIC_PROOF",
        "is_a_measurement": False,
        "occupancy_is_not_available_compute": True,
    }
    for name in HARDWARE_FIELDS:
        if name in obs and obs[name] is not None:
            raise HardwareClaimError(f"synthetic observation carried hardware field {name}")
        obs[name] = None
    refuse_hardware_tree(obs)
    return obs


def observe(
    concurrency: int,
    *,
    synthetic: Mapping[str, Any] | None = None,
    resident_sample: Mapping[str, Any] | None = None,
    capability: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Per-level observation. Absent resident or lease → REFUSED, not a fake pass."""
    if concurrency not in LEVELS:
        raise ObservationRefuse(f"concurrency {concurrency} is not on the ladder {LEVELS}")
    if synthetic is not None:
        fields = {k: v for k, v in dict(synthetic).items() if k != "concurrency"}
        return make_synthetic_observation(concurrency=concurrency, **fields)

    cap = dict(capability) if capability is not None else host_capability(
        resident_sample=resident_sample
    )
    presence = cap.get("resident_presence")
    slots = unknown_slots()
    if presence != PRESENCE_PRESENT:
        reason = (
            f"no resident process (presence={presence!r}); refusing a concurrency "
            "observation rather than inventing one"
        )
    else:
        reason = (
            "resident is PRESENT but this sidecar has no GPU lease; useful-work / "
            "token_ns / TTFT / dispatch / sync stay UNKNOWN"
        )
    out = {
        "status": "REFUSED",
        "source": "LIVE_REFUSED",
        "concurrency": concurrency,
        "reason": reason,
        "slots": slots,
        "capability_envelope": {
            "resident_bytes": cap.get("resident_bytes"),
            "memory_pressure": cap.get("memory_pressure"),
            "swap": cap.get("swap"),
            "gpu_occupancy_class": cap.get("gpu_occupancy_class"),
            "occupancy_is_not_available_compute": True,
        },
        "gpu_authority": False,
        "evidence_class": "STATIC_ONLY",
        "is_a_measurement": False,
        "verdict": None,
    }
    refuse_hardware_tree(out)
    return out


def public_observation(obs: Mapping[str, Any]) -> dict[str, Any]:
    """Receipt shape: classes and refusals, never useful-work magnitudes."""
    skip = {"useful_work_per_wall_second", "per_session_useful_work"}
    out = {k: v for k, v in obs.items() if k not in skip}
    out["useful_work_recorded_as_magnitude"] = False
    out["hardware_named_slots"] = {name: None for name in sorted(HARDWARE_FIELDS)}
    refuse_hardware_tree(out)
    return out


# ---------------------------------------------------------------------------
# advance / verdict
# ---------------------------------------------------------------------------


def _useful(obs: Mapping[str, Any]) -> float:
    value = obs.get("useful_work_per_wall_second")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ObservationRefuse(
            f"concurrency {obs.get('concurrency')}: useful_work_per_wall_second "
            f"is {value!r}; missing/zero is not a result"
        )
    return float(value)


def useful_work_delta_class(prev: Mapping[str, Any], cur: Mapping[str, Any]) -> str:
    a = _useful(prev)
    b = _useful(cur)
    if b * INFORMATIVE_DEN > a * INFORMATIVE_NUM:
        return "UP"
    if a * INFORMATIVE_DEN > b * INFORMATIVE_NUM:
        return "DOWN"
    return "FLAT"


def _is_refused(obs: Mapping[str, Any]) -> bool:
    return obs.get("status") == "REFUSED" or obs.get("source") in {"LIVE_REFUSED", "REFUSED"}


def advance(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Next concurrency, or STOP. Occupancy change is never a reason to continue."""
    if not observations:
        return {"next": 1, "action": "RUN", "why": "control level; always run"}
    ordered = sorted(observations, key=lambda o: int(o["concurrency"]))
    last = ordered[-1]
    last_n = int(last["concurrency"])
    if _is_refused(last):
        return {
            "next": None,
            "action": "STOP",
            "why": "last level refused; will not continue a failed ladder",
        }
    if last_n not in LEVELS:
        raise ObservationRefuse(f"concurrency {last_n} is not on the ladder")
    if last_n == LEVELS[-1]:
        return {
            "next": None,
            "action": "STOP",
            "why": "ladder end; concurrency 4 is the last probe this doctor will ask for",
        }
    if len(ordered) < 2:
        return {
            "next": 2,
            "action": "RUN",
            "why": "need concurrency 2 to compare useful work against the control",
        }
    prev = ordered[-2]
    delta = useful_work_delta_class(prev, last)
    if delta == "UP":
        nxt = last_n + 1
        return {
            "next": nxt,
            "action": "RUN",
            "why": (
                f"concurrency {last_n} was informative (useful work UP vs "
                f"{prev['concurrency']})"
            ),
            "delta_class": delta,
        }
    return {
        "next": None,
        "action": "STOP",
        "why": (
            f"concurrency {last_n} was not informative (useful work {delta} vs "
            f"{prev['concurrency']}); occupancy change is never a reason to continue"
        ),
        "delta_class": delta,
    }


def verdict(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Objective is useful work per wall second. Occupancy is never the ranking key."""
    if not observations:
        raise VerdictRefuse(
            "no observations; refusing a verdict rather than defaulting to CONCURRENCY_HELPS"
        )
    for obs in observations:
        refuse_hardware_tree(obs)
        if _is_refused(obs):
            raise VerdictRefuse(
                f"observation at concurrency {obs.get('concurrency')} is REFUSED "
                f"({obs.get('reason')}); no verdict"
            )
    ordered = sorted(observations, key=lambda o: int(o["concurrency"]))
    if len(ordered) < 2:
        raise VerdictRefuse(
            "need at least two concurrency levels of useful work; one level is a "
            "control, not a comparison"
        )
    if int(ordered[0]["concurrency"]) != 1:
        raise VerdictRefuse("ladder must include concurrency 1 as the control")

    works = [_useful(o) for o in ordered]
    winner_idx = max(
        range(len(works)),
        key=lambda i: (works[i], -int(ordered[i]["concurrency"])),
    )
    winner = ordered[winner_idx]
    control = ordered[0]
    deltas = []
    for prev, cur in zip(ordered, ordered[1:]):
        deltas.append(
            {
                "from": int(prev["concurrency"]),
                "to": int(cur["concurrency"]),
                "class": useful_work_delta_class(prev, cur),
            }
        )
    any_up = any(row["class"] == "UP" for row in deltas)
    gpu = control.get("gpu_occupancy_class")
    ceremony = control.get("host_ceremony_class")

    if any_up:
        name = "CONCURRENCY_HELPS"
        why = (
            f"useful work per wall second increased; occupancy is not the ranking "
            f"key (winner is concurrency {winner['concurrency']}, not the highest "
            f"occupancy class)"
        )
    elif gpu == "LOW" and ceremony == "HIGH":
        name = "HEADROOM_IS_HOST_CEREMONY"
        why = (
            "GPU occupancy is LOW while host ceremony is HIGH; the GPU idles on "
            "CPU prep/readback/sync. A second stream is the wrong lever; "
            "eliminate ceremony."
        )
    elif gpu in {None, "UNKNOWN"} or (
        gpu == "LOW" and ceremony in {None, "UNKNOWN"}
    ):
        raise VerdictRefuse(
            "useful work did not increase and ceremony/occupancy classes are "
            "UNKNOWN; refusing to guess between NO_USEFUL_CONCURRENCY_HEADROOM "
            "and HEADROOM_IS_HOST_CEREMONY"
        )
    else:
        name = "NO_USEFUL_CONCURRENCY_HEADROOM"
        extra = (
            "; GPU occupancy was not LOW, so this is not diagnosed as host ceremony"
            if gpu != "LOW"
            else "; host ceremony was not HIGH"
        )
        why = (
            "a second stream did not increase useful work per wall second" + extra
            + ". Bandwidth-bound / admission-bound: a second stream costs."
        )

    out = {
        "verdict": name,
        "why": why,
        "objective": "verified useful work per wall second, never utilisation",
        "winner_concurrency": int(winner["concurrency"]),
        "deltas": deltas,
        "gpu_occupancy_class_at_control": gpu,
        "host_ceremony_class_at_control": ceremony,
        "occupancy_is_not_the_objective": True,
        "high_occupancy_with_less_useful_work_loses": True,
        "evidence_class": (
            "SYNTHETIC_PROOF"
            if all(o.get("source") == "SYNTHETIC" for o in ordered)
            else "STATIC_ONLY"
        ),
        "gpu_authority": False,
        "is_a_measurement": False,
        "scope_required_to_persist": list(SCOPE_FIELDS),
        "does_not_universalise_to": list(UNIVERSAL_REFUSALS),
    }
    if name not in VERDICTS:
        raise VerdictRefuse(f"internal verdict {name!r} is not in {VERDICTS}")
    refuse_hardware_tree(out)
    return out


# ---------------------------------------------------------------------------
# scoped law
# ---------------------------------------------------------------------------


def seal_law(
    *,
    verdict_name: str,
    statement: str,
    machine: str,
    nx: str,
    runtime: str,
    context_regime: str,
    applies_to: Sequence[str] | None = None,
    transfer_to: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Persist a result only with machine + NX + runtime + context_regime.

    Authoritative is always false here: this sidecar did not run the ladder.
    """
    missing = [
        name
        for name, val in (
            ("machine", machine),
            ("nx", nx),
            ("runtime", runtime),
            ("context_regime", context_regime),
        )
        if not _present(val)
    ]
    if missing:
        raise ScopeRefused(f"law refused: missing/UNKNOWN scope field(s) {missing}")
    if context_regime not in CONTEXT_REGIMES:
        raise ScopeRefused(
            f"context_regime {context_regime!r} is not one of {CONTEXT_REGIMES}"
        )
    if verdict_name not in VERDICTS:
        raise VerdictRefuse(f"verdict {verdict_name!r} is not one of {VERDICTS}")
    if str(machine).strip().lower() in GENERIC_MACHINES:
        raise ScopeRefused(
            f"machine {machine!r} is a class, not this machine; scoping is the science"
        )
    extras = [str(x) for x in (*(applies_to or ()), *(transfer_to or ()))]
    blob = " ".join(extras).lower()
    for needle in UNIVERSAL_REFUSALS:
        if needle.lower() in blob:
            raise ScopeRefused(
                f"law refused: {needle} is a different machine/runtime/model; "
                f"this law is scoped to machine={machine!r} nx={nx!r} "
                f"runtime={runtime!r} context_regime={context_regime!r} and "
                f"does not transfer"
            )
    if not _present(statement):
        raise ScopeRefused("law refused: statement is empty")
    law = {
        "status": "SEALED_SCOPED",
        "authoritative": False,
        "verdict": verdict_name,
        "statement": statement,
        "scope": {
            "machine": str(machine),
            "nx": str(nx),
            "runtime": str(runtime),
            "context_regime": str(context_regime),
        },
        "does_not_apply_to": list(UNIVERSAL_REFUSALS),
        "evidence_class": "SYNTHETIC_PROOF",
        "gpu_authority": False,
        "is_a_measurement": False,
        "why_not_authoritative": (
            "this sidecar did not run the protected ladder; a scoped law record "
            "here is a shape proof, not a physical result"
        ),
    }
    refuse_hardware_tree(law)
    return law


# ---------------------------------------------------------------------------
# proofs that the three verdicts actually fire
# ---------------------------------------------------------------------------


def prove_synthetic() -> dict[str, Any]:
    """Drive every legal verdict from synthetic observations. A proof nobody
    watched fail is not a proof."""
    results: dict[str, Any] = {}

    helps = verdict(
        [
            make_synthetic_observation(
                concurrency=1,
                useful_work_per_wall_second=10.0,
                gpu_occupancy_class="MEDIUM",
                host_ceremony_class="LOW",
            ),
            make_synthetic_observation(
                concurrency=2,
                useful_work_per_wall_second=18.0,
                gpu_occupancy_class="HIGH",
                host_ceremony_class="LOW",
            ),
        ]
    )
    results["CONCURRENCY_HELPS"] = {
        "passed": helps["verdict"] == "CONCURRENCY_HELPS",
        "verdict": helps["verdict"],
        "winner_concurrency": helps["winner_concurrency"],
    }

    nohead = verdict(
        [
            make_synthetic_observation(
                concurrency=1,
                useful_work_per_wall_second=10.0,
                gpu_occupancy_class="HIGH",
                host_ceremony_class="LOW",
            ),
            make_synthetic_observation(
                concurrency=2,
                useful_work_per_wall_second=8.0,
                gpu_occupancy_class="HIGH",
                host_ceremony_class="LOW",
            ),
        ]
    )
    results["NO_USEFUL_CONCURRENCY_HEADROOM"] = {
        "passed": nohead["verdict"] == "NO_USEFUL_CONCURRENCY_HEADROOM",
        "verdict": nohead["verdict"],
    }

    ceremony = verdict(
        [
            make_synthetic_observation(
                concurrency=1,
                useful_work_per_wall_second=10.0,
                gpu_occupancy_class="LOW",
                host_ceremony_class="HIGH",
            ),
            make_synthetic_observation(
                concurrency=2,
                useful_work_per_wall_second=10.0,
                gpu_occupancy_class="LOW",
                host_ceremony_class="HIGH",
            ),
        ]
    )
    results["HEADROOM_IS_HOST_CEREMONY"] = {
        "passed": ceremony["verdict"] == "HEADROOM_IS_HOST_CEREMONY",
        "verdict": ceremony["verdict"],
    }

    loses = verdict(
        [
            make_synthetic_observation(
                concurrency=1,
                useful_work_per_wall_second=10.0,
                gpu_occupancy_class="MEDIUM",
                host_ceremony_class="LOW",
            ),
            make_synthetic_observation(
                concurrency=2,
                useful_work_per_wall_second=7.0,
                gpu_occupancy_class="HIGH",
                host_ceremony_class="LOW",
            ),
        ]
    )
    results["high_occupancy_with_less_useful_work_loses"] = {
        "passed": loses["verdict"] == "NO_USEFUL_CONCURRENCY_HEADROOM"
        and loses["verdict"] != "CONCURRENCY_HELPS",
        "verdict": loses["verdict"],
        "setup": "concurrency 2 HIGH occupancy, lower useful work than control",
    }

    try:
        verdict([])
        results["empty_observations_refuse"] = {"passed": False}
    except VerdictRefuse:
        results["empty_observations_refuse"] = {"passed": True}

    stub_cap = {
        "resident_presence": "UNDECLARED",
        "gpu_authority": False,
        "protected_lease": False,
        "quiescence": "UNKNOWN",
        "resident_bytes": None,
        "memory_pressure": "UNKNOWN",
        "swap": {"status": "UNKNOWN", "swap_ins": None},
        "gpu_occupancy_class": "UNKNOWN",
        "not_runnable_reasons": _not_runnable_reasons(
            {
                "resident_presence": "UNDECLARED",
                "gpu_authority": False,
                "protected_lease": False,
                "quiescence": "UNKNOWN",
            }
        ),
    }
    refused_obs = observe(1, capability=stub_cap)
    try:
        verdict([refused_obs])
        results["refused_observation_has_no_verdict"] = {"passed": False}
    except VerdictRefuse:
        results["refused_observation_has_no_verdict"] = {
            "passed": True,
            "observe_status": refused_obs["status"],
            "observe_verdict": refused_obs.get("verdict"),
        }

    decision = decide(stub_cap)
    results["undeclared_resident_is_sleeping"] = {
        "passed": decision["experiment_state"] == "SLEEPING"
        and decision["verdict"] is None
        and decision["verdict"] != "CONCURRENCY_HELPS",
        "experiment_state": decision["experiment_state"],
        "verdict": decision["verdict"],
    }

    try:
        refuse_hardware_named_number("accepted_tps", 12.0)
        results["hardware_named_field_raises"] = {"passed": False}
    except HardwareClaimError:
        results["hardware_named_field_raises"] = {"passed": True}

    try:
        seal_law(
            verdict_name="CONCURRENCY_HELPS",
            statement="unscoped",
            machine="Apple M3 Ultra",
            nx="",
            runtime="Metal",
            context_regime="WHOLE_MODEL_BODY",
        )
        results["law_without_nx_refused"] = {"passed": False}
    except ScopeRefused:
        results["law_without_nx_refused"] = {"passed": True}

    try:
        seal_law(
            verdict_name="CONCURRENCY_HELPS",
            statement="does not transfer",
            machine="Apple M3 Ultra",
            nx="noetic-sealed-3.14",
            runtime="Metal",
            context_regime="WHOLE_MODEL_BODY",
            applies_to=["Flash"],
        )
        results["law_flash_transfer_refused"] = {"passed": False}
    except ScopeRefused:
        results["law_flash_transfer_refused"] = {"passed": True}

    ok_law = seal_law(
        verdict_name="NO_USEFUL_CONCURRENCY_HEADROOM",
        statement=(
            "on this machine, this NX, this runtime, this context regime, a "
            "second stream did not increase useful work per wall second"
        ),
        machine="Apple M3 Ultra",
        nx="noetic-sealed-3.14",
        runtime="Metal",
        context_regime="WHOLE_MODEL_BODY",
    )
    results["scoped_law_shape"] = {
        "passed": ok_law["scope"]["context_regime"] == "WHOLE_MODEL_BODY"
        and ok_law["authoritative"] is False,
        "status": ok_law["status"],
        "authoritative": ok_law["authoritative"],
    }

    stopped = advance(
        [
            make_synthetic_observation(
                concurrency=1,
                useful_work_per_wall_second=10.0,
                gpu_occupancy_class="HIGH",
                host_ceremony_class="LOW",
            ),
            make_synthetic_observation(
                concurrency=2,
                useful_work_per_wall_second=10.0,
                gpu_occupancy_class="HIGH",
                host_ceremony_class="LOW",
            ),
        ]
    )
    results["flat_stops_the_ladder"] = {
        "passed": stopped["action"] == "STOP" and stopped["next"] is None,
        "action": stopped["action"],
    }

    results["all_passed"] = all(
        bool(row.get("passed")) for row in results.values() if isinstance(row, dict)
    )
    results["verdicts_reached"] = sorted(
        {helps["verdict"], nohead["verdict"], ceremony["verdict"]}
    )
    return results


# ---------------------------------------------------------------------------
# receipt
# ---------------------------------------------------------------------------


def recovered_implementation() -> list[str]:
    return [
        "tools/future/hardware_doctor.py metal_state — capability probe that is not a measurement (is_a_measurement false)",
        "tools/future/metal_reachability.py — device enumeration, not a kernel timing",
        "tools/future/contamination.py probe_memory / probe_gpu_occupancy — machine state; occupancy is not available compute; HEAVY_GPU_UTIL_PCT reused as the LOW/not-LOW boundary",
        "tools/future/resident_health.py sample — UNDECLARED/ABSENT with rss_bytes null, never a pid invented from the largest RSS neighbour",
        "tools/future/resident_identity.py — machine_genome / nx_id / current_backend slots a scoped law must name",
        "tools/future/hcli_self_profile.py assert_timing_field_legal — hardware-named fields raise even for a Python timing",
        "tools/future/_common.py HARDWARE_FIELDS / write_receipt — numeric tps/token_ns/gpu_ns raise",
        "tools/future/workunit_species.py emit_hcli_workunit — SLEEPING is blocked, not a synthetic COMPLETED",
        "tools/future/frontiers.py THIS_HOST_LANES / HARDWARE_LANES / _emit_unit(sleeping=True) — GPU_PROTECTED stays SLEEPING until disk evidence qualifies",
        "tools/future/wakeup.py SLEEPING WorkUnit wake_condition shape",
        "tools/future/odyssey2_law_store.py — sequential scope lattice; this module does not fork it, it refuses an unscoped concurrency law",
        "tools/future/qwen27_profile_schema.py host_ceremony organ — the ceremony bucket the HEADROOM_IS_HOST_CEREMONY verdict names",
        "tools/future/decode_civilization.py accepted_complete_token_cost — useful work, not raw draft throughput",
        "receipts/headless/ACCELERATOR_CONCURRENCY_SWEEP.json — 4 MiB kernel vs whole-body regimes; cited as a qualitative warning, numbers not copied",
        "hawking-experiments/superwave/g1/g1-worker-concurrency.md — four sessions, decode_concurrency=1; cited as a qualitative warning, numbers not copied",
    ]


def gaps_closed() -> list[str]:
    return [
        "plan() for resident-session concurrency 1,2,3,4 continuing only while useful work is informative",
        "observe() per level with refused UNKNOWN slots when no resident / no lease",
        "verdict() on verified useful work per wall second; occupancy is not the ranking key",
        "CONCURRENCY_HELPS, NO_USEFUL_CONCURRENCY_HEADROOM, HEADROOM_IS_HOST_CEREMONY all reachable from synthetic observations",
        "SLEEPING WorkUnit with wake condition when the ladder cannot run",
        "seal_law requires machine, NX, runtime, context_regime and refuses Flash/M5/FPGA/CUDA transfer",
        "hardware-named numeric fields raise rather than seal",
    ]


def negative_findings(capability: Mapping[str, Any], proofs: Mapping[str, Any]) -> list[str]:
    findings = [
        "this sidecar did not start a resident, did not take a GPU lease, and did not run the concurrency ladder",
        "useful-work, token_ns, TTFT, dispatch rate and synchronisation stay UNKNOWN; they are not defaulted",
        "partial GPU occupancy, if probed, is machine state and a hypothesis about headroom, not available compute",
        "the recovered ACCELERATOR_CONCURRENCY_SWEEP is a 4 MiB kernel in SMALL_KERNEL_LAUNCH_BOUND; citing it as a whole-body law is the exact mixing this scoping refuses",
        "orchestration BINDINGS / frontiers catalog were outside this lane's WRITE list, so this module is not glued into invoke() by this receipt",
        "host_ceremony_class cannot be observed live without token_ns / dispatch_ns this sidecar will not record",
        "a doctor that cannot run today is still worth building; a doctor that fabricates a TPS is worse than none",
    ]
    presence = capability.get("resident_presence")
    if presence != PRESENCE_PRESENT:
        findings.append(
            f"resident presence at build is {presence!r}; the experiment is SLEEPING"
        )
    if not proofs.get("all_passed"):
        findings.append("synthetic proofs did not all pass; the receipt must not have been written")
    return findings


def build() -> Path:
    proofs = prove_synthetic()
    if not proofs.get("all_passed"):
        raise DoctorRefuse(f"synthetic negative controls did not all fire: {proofs}")
    cap = host_capability()
    decision = decide(cap)
    planned = plan()
    live_obs = [public_observation(observe(n, capability=cap)) for n in LEVELS]
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Plan and (when a resident and a protected lease exist) settle whether "
            "resident-session concurrency increases verified useful work per wall "
            "second. This receipt is the plan and the host's capability to run it, "
            "not a measurement."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "is_a_measurement": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "objective": planned["objective"],
        "vocabulary": {
            "verdicts": list(VERDICTS),
            "occupancy_classes": list(OCCUPANCY_CLASSES),
            "ceremony_classes": list(CEREMONY_CLASSES),
            "delta_classes": list(DELTA_CLASSES),
            "context_regimes": list(CONTEXT_REGIMES),
            "scope_fields": list(SCOPE_FIELDS),
            "eras": list(ERAS),
            "odysseys": list(ODYSSEYS),
        },
        "plan": planned,
        "host_capability": cap,
        "experiment_state": decision["experiment_state"],
        "verdict": decision["verdict"],
        "decision": {
            "experiment_state": decision["experiment_state"],
            "verdict": decision["verdict"],
            "reason": decision["reason"],
            "resident_presence": decision["resident_presence"],
            "wake_condition": decision["wake_condition"],
        },
        "sleeping_workunit": decision["workunit"],
        "live_observations": live_obs,
        "law": {
            "status": "NOT_SEALED",
            "reason": (
                "no protected observations; a law without evidence would be fiction"
            ),
            "required_scope": list(SCOPE_FIELDS),
            "refuses_to_universalise": list(UNIVERSAL_REFUSALS),
            "authoritative": False,
        },
        "proofs": {
            "synthetic": {
                "all_passed": proofs["all_passed"],
                "verdicts_reached": proofs["verdicts_reached"],
                "controls": {
                    name: {
                        k: v
                        for k, v in row.items()
                        if k != "setup" or not isinstance(v, (int, float))
                    }
                    for name, row in proofs.items()
                    if isinstance(row, dict) and name != "all_passed"
                },
            }
        },
        "recovered_prior_science": [
            {
                "path": "receipts/headless/ACCELERATOR_CONCURRENCY_SWEEP.json",
                "what": (
                    "4 MiB kernel launch-bound sweep vs whole-body bandwidth-bound "
                    "law; different context_regime; this lane did not re-measure "
                    "and will not cite it as a resident-session law"
                ),
                "this_lane_remeasured": False,
                "numbers_copied": False,
            },
            {
                "path": "hawking-experiments/superwave/g1/g1-worker-concurrency.md",
                "what": (
                    "one body, four sessions, decode_concurrency=1; concurrent "
                    "decode recorded as a negative for aggregate useful work; "
                    "this lane did not re-measure"
                ),
                "this_lane_remeasured": False,
                "numbers_copied": False,
            },
        ],
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(cap, proofs),
        "resident_callable": {
            "entry_point": "tools.future.concurrency_doctor.plan() / observe() / verdict()",
            "workunit": (
                "one CPU_ANALYSIS unit for the plan; the live experiment is "
                "GPU_EXCLUSIVE and SLEEPING until resident+lease+quiescence"
            ),
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": FRONTIER,
            "fails_closed": (
                "absent resident or absent lease emits SLEEPING and refuses a "
                "verdict; never defaults to CONCURRENCY_HELPS; hardware-named "
                "numbers raise; an unscoped law raises"
            ),
            "orchestration_bound": False,
        },
    }
    refuse_hardware_tree(doc)
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    return build()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.parse_args()
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
