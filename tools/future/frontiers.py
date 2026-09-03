"""FRONTIER_STATE — twenty-two persistent frontiers a resident cannot idle past.

global_frontier.py is the campaign gap tracker (F001–F020, probe-backed).
It is adequate for that job and is not forked here. This module is the
operating layer on top: named frontiers with persistent open questions,
SLEEPING blocked work (wake conditions, never a synthetic result),
next_work() for the lanes that are actually available, busywork refusal
at admission, and a movement metric that stays UNKNOWN until a verified
non-dominated move exists.

    python3 tools/future/frontiers.py --build
    python3 tools/future/frontiers.py --selftest
    python3 tools/future/frontiers.py --next-work --lanes CPU,SIMULATION,REPRESENTATION,TOOLING,ODYSSEY,ANALYSIS

HCLI invoke path: next_work / is_idle / admit / build. WorkUnits are
emitted through the landed workunit_species constructor. Concurrent-wave
modules (resident_api, workgraph, wakeup, super_resident, …) are not
imported; they are named as integration swaps in the receipt.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO, git, HARDWARE_FIELDS, _assert_no_hardware_claims

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools.future import negative_index as ni
from tools.future import workunit_species as wus


RECEIPT = "FRONTIER_STATE.json"
SCHEMA = "hawking.future.frontiers.v1"
RECORDED_BY = "tools/future/frontiers.py"
HANDOFF_REL = "CODEX_ACCELERATOR_HANDOFF.json"
GLOBAL_FRONTIER_REL = "receipts/future/CLAUDE_GLOBAL_FRONTIER.json"
QUEUE_REL = "receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json"
SCAR_INDEX_REL = "receipts/future/NEGATIVE_SCIENCE_INDEX.json"

# Canonical operating frontiers. The set is frozen by the campaign; counts
# of *items inside* each frontier are derived from disk, never hard-coded.
FRONTIER_NAMES: tuple[str, ...] = (
    "MODEL_REPRESENTATION",
    "MODEL_CAPABILITY",
    "MODEL_EXECUTION",
    "LATENCY",
    "TPS",
    "ACTIVE_BYTES",
    "GPU_KERNELS",
    "STATE",
    "DECODING",
    "HCLI_SELF",
    "EXPERIMENT_TURNAROUND",
    "PHYSICAL_GRAPH",
    "FPGA",
    "ANE",
    "ARCHITECTURE_REPATRIATION",
    "TOOLS",
    "CONTEXT",
    "MEMORY",
    "VERIFICATION",
    "ODYSSEY_TRANSFER",
    "ODYSSEY_ADVERSARY",
    "CHILD_RESIDENT",
)

# Lanes a caller may name. Hardware lanes stay SLEEPING until disk evidence
# says they have qualified; listing the name is not a qualification.
LANE_CPU = "CPU"
LANE_ANALYSIS = "ANALYSIS"
LANE_SIMULATION = "SIMULATION"
LANE_REPRESENTATION = "REPRESENTATION"
LANE_TOOLING = "TOOLING"
LANE_ODYSSEY = "ODYSSEY"
LANE_GPU_PROTECTED = "GPU_PROTECTED"
LANE_ANE = "ANE"
LANE_FPGA = "FPGA"

CPU_LANES: tuple[str, ...] = (
    LANE_CPU,
    LANE_ANALYSIS,
    LANE_SIMULATION,
    LANE_REPRESENTATION,
    LANE_TOOLING,
    LANE_ODYSSEY,
)
HARDWARE_LANES: tuple[str, ...] = (LANE_GPU_PROTECTED, LANE_ANE, LANE_FPGA)
ALL_LANES: tuple[str, ...] = CPU_LANES + HARDWARE_LANES

# This sidecar host: CPU-class lanes only. GPU_PROTECTED and ANE are blocked
# (Codex's own list). FPGA board is not here; FPGA *simulation* uses SIMULATION.
THIS_HOST_LANES: tuple[str, ...] = CPU_LANES
BLOCKED_ON_THIS_HOST: tuple[str, ...] = HARDWARE_LANES

ERAS = ("I", "II", "III", "IV", "V")
ODYSSEYS = ("I", "II", "III")

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement. Does not produce "
    "DIAGNOSTIC_RELATIVE or PROTECTED_ABSOLUTE. SLEEPING physical work is "
    "never filled in with a synthetic result."
)
PROPOSAL_CLAIM_BOUNDARY = wus.PROPOSAL_CLAIM_BOUNDARY

INFO_HIGH, INFO_MEDIUM, INFO_LOW, INFO_NONE = 3, 2, 1, 0
REDUNDANCY_EXACT = 0.85
REDUNDANCY_LOW_GAIN = 0.50

# Campaign-gap tracker entries mapped onto operating frontiers. Resolved
# sidecar closures stay as history, not as open work.
F_TO_FRONTIERS: dict[str, tuple[str, ...]] = {
    "F001": ("GPU_KERNELS", "MODEL_EXECUTION", "STATE"),
    "F002": ("GPU_KERNELS", "LATENCY", "TPS"),
    "F003": ("PHYSICAL_GRAPH", "FPGA"),
    "F004": ("FPGA", "PHYSICAL_GRAPH"),
    "F005": ("MEMORY", "ACTIVE_BYTES"),
    "F006": ("VERIFICATION",),
    "F007": ("MODEL_REPRESENTATION", "TOOLS"),
    "F008": ("VERIFICATION",),
    "F009": ("VERIFICATION", "TOOLS"),
    "F010": ("ODYSSEY_TRANSFER",),
    "F011": ("VERIFICATION",),
    "F012": ("PHYSICAL_GRAPH", "ARCHITECTURE_REPATRIATION"),
    "F013": ("MODEL_CAPABILITY", "CHILD_RESIDENT"),
    "F014": ("GPU_KERNELS", "VERIFICATION"),
    "F015": ("TOOLS", "ODYSSEY_TRANSFER"),
    "F016": ("TOOLS", "ODYSSEY_TRANSFER", "ODYSSEY_ADVERSARY"),
    "F017": ("TOOLS",),
    "F018": ("GPU_KERNELS", "VERIFICATION"),
    "F019": ("MODEL_REPRESENTATION", "MODEL_CAPABILITY", "STATE"),
    "F020": ("PHYSICAL_GRAPH", "FPGA", "ARCHITECTURE_REPATRIATION"),
}

_STOP = frozenset(
    """
    the a an and or of to in for on with without per from into by is are be this
    that if only remain remains remaining unchanged no none not measure measured
    while when where which whose their its one two plus than more less over under
    as at it we a unit work next open blocked sleeping
    """.split()
)
_WORD = re.compile(r"[a-z0-9]+")


class UnverifiedMoveError(ValueError):
    """A frontier move was offered without verification evidence."""


class DominatedMoveError(ValueError):
    """The offered after-point does not strictly non-dominate the before-point."""


class FabricatedBaselineError(ValueError):
    """Seeding movement with an invented baseline is refused."""


class AdmissionRefused(ValueError):
    """A proposal was refused at admission (busywork, scar, or zero gain)."""


# ---------------------------------------------------------------------------
# Recovery. Missing in this sparse tree is a path taken, not a project-absent.
# ---------------------------------------------------------------------------

def _checkout_roots() -> list[Path]:
    roots: list[Path] = [REPO]
    blob = git("worktree", "list", "--porcelain")
    for line in blob.splitlines():
        if line.startswith("worktree "):
            roots.append(Path(line.split(" ", 1)[1]))
    common = git("rev-parse", "--git-common-dir")
    if common:
        path = Path(common)
        if not path.is_absolute():
            path = (REPO / path).resolve()
        else:
            path = path.resolve()
        parent = path.parent if path.name == ".git" else path.parent
        if parent not in roots:
            roots.append(parent)
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            key = str(root.resolve())
        except OSError:
            key = str(root)
        if key in seen:
            continue
        seen.add(key)
        out.append(root)
    return out


def load_optional(rel: str) -> tuple[dict[str, Any] | None, str]:
    """Load JSON. Unseen in this checkout is not evidence the file does not exist."""
    searched: list[str] = []
    for root in _checkout_roots():
        path = root / rel
        searched.append(str(path))
        if path.is_file():
            try:
                data = load_json(path)
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                return None, f"unreadable:{path}:{exc}"
            if isinstance(data, dict):
                return data, f"disk:{path}"
            return None, f"not_object:{path}"
    blob = git("show", f"HEAD:{rel}")
    if blob:
        try:
            data = json.loads(blob)
        except json.JSONDecodeError as exc:
            return None, f"git_unreadable:HEAD:{rel}:{exc}"
        if isinstance(data, dict):
            return data, f"git:HEAD:{rel}"
    return None, "unseen_in_this_checkout"


def _future_receipt(name: str) -> tuple[dict[str, Any] | None, str]:
    return load_optional(f"receipts/future/{name}")


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_WORD.findall((text or "").lower())) - _STOP


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s or "item"


def _lanes(available_lanes: Iterable[str] | str | None) -> frozenset[str]:
    if available_lanes is None:
        return frozenset(THIS_HOST_LANES)
    if isinstance(available_lanes, str):
        parts = [p.strip() for p in available_lanes.split(",") if p.strip()]
        return frozenset(parts)
    return frozenset(str(x) for x in available_lanes)


def _redundancy_key(item: Mapping[str, Any]) -> str:
    if item.get("redundancy_key"):
        return str(item["redundancy_key"])
    frontier = str(item.get("frontier") or "")
    family = str(item.get("hypothesis_family") or "")
    cand = str(item.get("candidate_id") or "")
    title = _slug(str(item.get("title") or item.get("id") or ""))
    return "|".join((frontier, family or title, cand))


# ---------------------------------------------------------------------------
# Catalog. Items are templates; live counts and wake evidence overlay at load.
# ---------------------------------------------------------------------------


# S026 §4 and TOP_LEVEL_TOKEN_REORDERING_HAS_NO_CURRENT_SLACK. G082 measured all
# 11 edges of the single-token data flow as TRUE dependencies with zero
# overlapable top-level work, so command-order permutations cannot reach the
# capacity multi-session execution exposes. A unit proposing one is refused HERE,
# at proposal time, rather than discovered after it has run.
#
# This refuses a SCHOOL, not a word. OPEN_QUESTIONs are exempt: asking whether
# the scar still holds is legitimate, proposing work that assumes it does not is
# what costs GPU time.
DEAD_SCHOOL_REORDERING = (
    "reorder",
    "reordering",
    "permute the dispatch",
    "dispatch permutation",
    "command-order",
    "command order permutation",
    "overlap top-level",
    "top-level overlap",
)


class DeadSchoolRefused(ValueError):
    """A proposed unit belongs to a school an emitted scar has closed."""


def _refuse_dead_school(
    id: str, kind: str, title: str, detail: str, family: str
) -> None:
    if kind == "OPEN_QUESTION":
        return
    hay = f"{title} {detail} {family}".lower()
    hit = next((p for p in DEAD_SCHOOL_REORDERING if p in hay), None)
    if hit is None:
        return
    raise DeadSchoolRefused(
        f"{id}: proposes {hit!r}, which belongs to the school closed by "
        "TOP_LEVEL_TOKEN_REORDERING_HAS_NO_CURRENT_SLACK "
        "(receipts/future/SINGLE_TOKEN_DAG.json, S026 §4): all 11 token edges "
        "are true dependencies and there is zero overlapable top-level work. "
        "The next questions live INSIDE kernels - occupancy, memory-level "
        "parallelism, register pressure, unpack and convert cost. If you "
        "believe the graph has changed, rebuild single_token_dag first: it "
        "recomputes the slack and refuses to emit the scar if any appears."
    )


def _item(
    *,
    id: str,
    frontier: str,
    kind: str,
    title: str,
    detail: str,
    required_lanes: Sequence[str],
    gain: int,
    species: str,
    verifier: str,
    evidence: Sequence[str],
    hypothesis_family: str = "",
    resource_class: str = "STATIC_ANALYSIS",
    effect_class: str = "READ_ONLY",
    wake_all_of: Sequence[str] = (),
    wake_never: Sequence[str] = (),
    candidate_id: str = "",
    source_f: str = "",
) -> dict[str, Any]:
    if frontier not in FRONTIER_NAMES:
        raise ValueError(f"unknown frontier {frontier!r}")
    kind_n = str(kind).strip().upper()
    if kind_n not in {"NEXT_WORK", "BLOCKED", "OPEN_QUESTION"}:
        raise ValueError(f"{id}: kind {kind!r} is not NEXT_WORK/BLOCKED/OPEN_QUESTION")
    _refuse_dead_school(id, kind_n, title, detail, hypothesis_family)
    lanes = tuple(str(x) for x in required_lanes)
    unknown = [x for x in lanes if x not in ALL_LANES]
    if unknown:
        raise ValueError(f"{id}: unknown lanes {unknown}")
    rec = {
        "id": id,
        "frontier": frontier,
        "kind": kind_n,
        "title": title,
        "detail": detail,
        "required_lanes": list(lanes),
        "expected_information_gain": int(gain),
        "species": species,
        "verifier": verifier,
        "evidence": list(evidence),
        "hypothesis_family": hypothesis_family or _slug(title).replace("-", "_"),
        "resource_class": resource_class,
        "effect_class": effect_class,
        "wake_all_of": list(wake_all_of),
        "wake_never": list(wake_never)
        or (
            [
                "synthetic result",
                "lease seizure / flock",
                "quiesce standing workers",
            ]
            if kind_n == "BLOCKED"
            else []
        ),
        "candidate_id": candidate_id,
        "source_f": source_f,
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
    }
    rec["redundancy_key"] = _redundancy_key(rec)
    return rec


_WAKE_GPU = (
    "Metal-capable GPU present on the execution host (MetalContext)",
    "xcrun locates the Metal compiler (not CommandLineTools-missing)",
    "existing HCLI protected lease with a proven holder pid (read, never flock)",
    "qualification pipeline classifies the machine QUIESCENT (assessed, never coerced)",
)
_WAKE_ANE = (
    "Core ML / ANE compiler environment available",
    "Flash-shaped ANE execution is authorized on this host",
)
_WAKE_FPGA = (
    "U50 / FPGA board present and the physical-compiler/fusion lane owns it",
)
_WAKE_NX = (
    "Flash source-independent NX is qualified (not SCAFFOLD_ONLY / SEALED_METADATA_ONLY)",
)
_WAKE_TEACHER = (
    "Metal-capable GPU for dense source-BF16 prefix initialization",
    "teacher capture rows meet the minimum (disk count, never a synthetic corpus)",
)
_WAKE_NEVER = (
    "synthetic result",
    "lease seizure / flock",
    "quiesce standing workers",
    "invented hardware number",
)


def _catalog() -> tuple[dict[str, Any], ...]:
    """CPU-safe next work, hardware SLEEPING units, and open questions.

    One unit per idea. READY_PROTECTED candidates collapse to a single
    SLEEPING unit whose identity set is filled from disk at load time —
    emitting one unit per candidate while they share a wake condition is
    the busywork this module exists to refuse.
    """
    ev = "receipts/future"
    hd = "receipts/headless"
    return (
        _item(
            # Distinct from the OPEN_QUESTION already on this frontier. Reusing
            # that id put two items with one id in the book, which the freeze
            # manifest caught as a duplicate. An open question and the work that
            # answers it are different items.
            id="FT.MODEL_CAPABILITY.hard-gates.drive-tools",
            frontier="MODEL_CAPABILITY",
            kind="NEXT_WORK",
            title="Drive the Odyssey Doctor and Gravity tools from the resident and route their receipts",
            detail=(
                "This frontier carried only an OPEN_QUESTION, so refill could "
                "never yield work on it and doctor_callable / gravity_callable "
                "were structurally unreachable. That is no longer the state: "
                "odyssey_tool_driver.py invokes doctor_seal for real and routes "
                "the receipt, and the remaining tools have named, checkable "
                "blockers rather than unknown ones -- doctor_tournament needs "
                "torch importable in the resident interpreter for its 52GB SVD, "
                "and the gravity mains need prior measurement files that are "
                "absent and must not be invented. The pending CPU work is to "
                "drive what can be driven and record the refusal for the rest."
            ),
            required_lanes=(LANE_CPU, LANE_ANALYSIS),
            gain=INFO_HIGH,
            species="odyssey_tool_invocation",
            verifier="future.odyssey_tool_driver.invoke",
            evidence=(f"{ev}/ODYSSEY_TOOL_DRIVER.json", f"{ev}/ODYSSEY_LAUNCH_GATE.json"),
            hypothesis_family="resident_tool_operation",
            source_f="F007",
        ),
        _item(
            id="FT.MODEL_REPRESENTATION.meta-gates-3-9",
            frontier="MODEL_REPRESENTATION",
            kind="NEXT_WORK",
            title="Prepare meta funnel gates 3-9 so they can start the moment teacher rows arrive",
            detail=(
                "F019 stalls all nine real Flash meta families at gate 2 "
                "(real_teacher_fit). meta_ready.py already writes the dossier; "
                "the remaining CPU work is to keep gates 3-9 wired to the "
                "corpus-arrival contract without fabricating teacher rows."
            ),
            required_lanes=(LANE_CPU, LANE_REPRESENTATION),
            gain=INFO_HIGH,
            species="learned_compiler_experiment",
            verifier="future.meta_ready.gates_3_9",
            evidence=(f"{ev}/META_DOWNSTREAM_READY.json", f"{ev}/META_EXPERIMENT_FUNNEL.json"),
            hypothesis_family="meta_funnel_readiness",
            source_f="F019",
        ),
        _item(
            id="FT.MODEL_REPRESENTATION.ngram-school",
            frontier="MODEL_REPRESENTATION",
            kind="NEXT_WORK",
            title="Generate n-gram-school representation candidates below Q4 without fitting weights",
            detail="ngram_school.py is executable; the next unit is a fresh candidate set scored against the negative index, not a GPU fit.",
            required_lanes=(LANE_CPU, LANE_REPRESENTATION),
            gain=INFO_MEDIUM,
            species="learned_compiler_experiment",
            verifier="future.ngram_school.candidates",
            evidence=(f"{ev}/NGRAM_SCHOOL.json", f"{ev}/NEGATIVE_SCIENCE_INDEX.json"),
            hypothesis_family="ngram_school_candidates",
        ),
        _item(
            id="FT.MODEL_REPRESENTATION.teacher-capture",
            frontier="MODEL_REPRESENTATION",
            kind="BLOCKED",
            title="Flash teacher capture is 0/256 and blocked on Metal",
            detail="FLASH_META_TEACHER_L4_CAPTURE_BOUNDARY status BLOCKED_NO_METAL_GPU at dense_source_bf16_prefix_initialization. Sleeps. No synthetic rows. The STATUS is not the CAUSE: METAL_REACHABILITY.json shows this host's GPU is reachable from the same metal crate, so the wake condition is identifying that process's context, not acquiring hardware.",
            required_lanes=(LANE_GPU_PROTECTED,),
            gain=INFO_HIGH,
            species="learned_compiler_experiment",
            verifier="future.teacher_corpus.capture",
            evidence=(f"{hd}/FLASH_META_TEACHER_L4_CAPTURE_BOUNDARY.json", f"{ev}/TEACHER_CORPUS_CONTRACT.json"),
            hypothesis_family="teacher_capture",
            resource_class="GPU_EXCLUSIVE",
            wake_all_of=_WAKE_TEACHER,
            source_f="F019",
        ),
        _item(
            id="FT.MODEL_CAPABILITY.tournament-refuse",
            frontier="MODEL_CAPABILITY",
            kind="NEXT_WORK",
            title="Keep the NX-vs-NX tournament harness in refuse-to-run until both contenders are complete",
            detail="F013: FLASH_SINGULARITY.NX vs QWEN27_SINGULARITY.NX. The harness must refuse, not invent a winner. CPU work: re-seal readiness against current NX seals.",
            required_lanes=(LANE_CPU, LANE_ANALYSIS),
            gain=INFO_MEDIUM,
            species="independent_reproduction",
            verifier="future.tournament.refuse_until_complete",
            evidence=(f"{ev}/TOURNAMENT_READINESS.json", f"{ev}/FLASH_NX_COMPLETENESS_AUDIT.json"),
            hypothesis_family="tournament_refuse_to_run",
            source_f="F013",
        ),
        _item(
            id="FT.MODEL_CAPABILITY.hard-gates",
            frontier="MODEL_CAPABILITY",
            kind="OPEN_QUESTION",
            title="Which capability hard-gates can be decided from sealed receipts without a protected window?",
            detail="Tournament hard gates cite capability_suite and resident_gate. Inventory which items are receipt-decidable vs GPU-bound.",
            required_lanes=(LANE_CPU, LANE_ANALYSIS),
            gain=INFO_MEDIUM,
            species="independent_reproduction",
            verifier="future.tournament.hard_gate_inventory",
            evidence=(f"{ev}/TOURNAMENT_READINESS.json",),
            hypothesis_family="capability_hard_gates",
        ),
        _item(
            id="FT.MODEL_EXECUTION.fusion-sim",
            frontier="MODEL_EXECUTION",
            kind="NEXT_WORK",
            title="CPU-simulate fusion_planner / fusion_isa graphs; graph shape is not a timing result",
            detail="fusion_simulation.py is executable. Run the simulation receipt refresh; do not claim token_ns.",
            required_lanes=(LANE_CPU, LANE_SIMULATION),
            gain=INFO_MEDIUM,
            species="fusion_simulation",
            verifier="future.fusion.simulate",
            evidence=(f"{ev}/FUSION_SIMULATION.json",),
            hypothesis_family="fusion_isa_graph",
            resource_class="COMPILE",
            effect_class="REVERSIBLE",
        ),
        _item(
            id="FT.MODEL_EXECUTION.static-skeleton",
            frontier="MODEL_EXECUTION",
            kind="NEXT_WORK",
            title="Validate static decode skeletons so slot-as-topology remains refused",
            detail="STATIC_SKELETON.json already names backend usability. Next: re-run the validator against current organ maps.",
            required_lanes=(LANE_CPU, LANE_ANALYSIS),
            gain=INFO_MEDIUM,
            species="independent_reproduction",
            verifier="future.static_skeleton.validate",
            evidence=(f"{ev}/STATIC_SKELETON.json",),
            hypothesis_family="static_skeleton_slots",
        ),
        _item(
            id="FT.MODEL_EXECUTION.complete-token",
            frontier="MODEL_EXECUTION",
            kind="BLOCKED",
            title="Protected complete-token execution is blocked on Metal + NX",
            detail="Flash critical path WAITING_FOR_COMPLETE_TOKEN; source-independent NX is SCAFFOLD_ONLY. Sleeps.",
            required_lanes=(LANE_GPU_PROTECTED,),
            gain=INFO_HIGH,
            species="accelerator_candidate_qualification",
            verifier="accelerator.physical.complete_token",
            evidence=(f"{ev}/FLASH_NX_COMPLETENESS_AUDIT.json",),
            hypothesis_family="complete_token_protected",
            resource_class="GPU_EXCLUSIVE",
            wake_all_of=_WAKE_GPU + _WAKE_NX,
            source_f="F001",
        ),
        _item(
            id="FT.LATENCY.cpu-turnaround",
            frontier="LATENCY",
            kind="NEXT_WORK",
            title="Remeasure CPU-side experiment-turnaround phases; leave GPU phases UNKNOWN",
            detail="turnaround.py already times source_discovery/transform/verify/receipt/ledger/next_decision. Refresh. cargo/GPU phases stay UNKNOWN.",
            required_lanes=(LANE_CPU, LANE_TOOLING),
            gain=INFO_MEDIUM,
            species="independent_reproduction",
            verifier="future.turnaround.cpu_phases",
            evidence=(f"{ev}/EXPERIMENT_TURNAROUND.json",),
            hypothesis_family="cpu_turnaround_phases",
        ),
        _item(
            id="FT.LATENCY.gpu-ns",
            frontier="LATENCY",
            kind="BLOCKED",
            title="Protected kernel/token latency is blocked on GPU_PROTECTED",
            detail="No Metal-capable GPU; xcrun has no Metal compiler under CommandLineTools. Sleeps. UNKNOWN is the correct latency answer.",
            required_lanes=(LANE_GPU_PROTECTED,),
            gain=INFO_HIGH,
            species="accelerator_candidate_qualification",
            verifier="accelerator.physical.latency",
            evidence=(HANDOFF_REL, f"{ev}/HARDWARE_DOCTOR.json"),
            hypothesis_family="protected_latency",
            resource_class="GPU_EXCLUSIVE",
            wake_all_of=_WAKE_GPU,
            source_f="F002",
        ),
        _item(
            id="FT.TPS.accepted-token-cost",
            frontier="TPS",
            kind="NEXT_WORK",
            title="Refresh accepted_complete_token_cost in relative units (not a TPS number)",
            detail="decode_civilization.py models accepted-token cost with rollback in the numerator. Relative units only; accepted_tps stays UNKNOWN.",
            required_lanes=(LANE_CPU, LANE_ANALYSIS),
            gain=INFO_HIGH,
            species="independent_reproduction",
            verifier="future.decode_civilization.accepted_token_cost",
            evidence=(f"{ev}/DECODE_CIVILIZATION.json",),
            hypothesis_family="accepted_complete_token_cost",
        ),
        _item(
            id="FT.TPS.protected-tps",
            frontier="TPS",
            kind="BLOCKED",
            title="Protected accepted-token TPS is blocked; Flash has one accepted stateful token",
            detail="FLASH_STATEFUL_TPS_GATE_V14 is BLOCKED_FIRST_PHYSICAL_BOUNDARY. Continuation state, repeated decode, protected TPS remain open. Sleeps.",
            required_lanes=(LANE_GPU_PROTECTED,),
            gain=INFO_HIGH,
            species="accelerator_candidate_qualification",
            verifier="accelerator.physical.accepted_tps",
            evidence=(f"{hd}/FLASH_STATEFUL_TPS_GATE_V14.json", HANDOFF_REL),
            hypothesis_family="protected_accepted_tps",
            resource_class="GPU_EXCLUSIVE",
            wake_all_of=_WAKE_GPU + _WAKE_NX,
            source_f="F002",
        ),
        _item(
            id="FT.ACTIVE_BYTES.hbm-rank",
            frontier="ACTIVE_BYTES",
            kind="NEXT_WORK",
            title="Rank HBM residency candidates from organ census; meter fields stay UNKNOWN",
            detail="hbm_doctor.py decides what 8 GB of HBM is for. CPU ranking of ResidentCandidate records; no invented bandwidth_gbps.",
            required_lanes=(LANE_CPU, LANE_ANALYSIS),
            gain=INFO_HIGH,
            species="hardware_doctor_experiment",
            verifier="future.hbm_doctor.solve",
            evidence=(f"{ev}/HBM_DOCTOR.json",),
            hypothesis_family="hbm_residency_rank",
            source_f="F005",
        ),
        _item(
            id="FT.ACTIVE_BYTES.measured",
            frontier="ACTIVE_BYTES",
            kind="BLOCKED",
            title="Measured active-byte / bandwidth figures require a protected window",
            detail="Sidecar has no GPU authority. Active-byte measurement sleeps.",
            required_lanes=(LANE_GPU_PROTECTED,),
            gain=INFO_MEDIUM,
            species="accelerator_candidate_qualification",
            verifier="accelerator.physical.active_bytes",
            evidence=(f"{ev}/HBM_DOCTOR.json",),
            hypothesis_family="measured_active_bytes",
            resource_class="GPU_EXCLUSIVE",
            wake_all_of=_WAKE_GPU,
        ),
        _item(
            id="FT.GPU_KERNELS.static-warnings",
            frontier="GPU_KERNELS",
            kind="NEXT_WORK",
            title="Triage remaining static kernel WARNINGs; do not re-raise adjudicated ERRORs",
            detail="F018 ABI findings were adjudicated. Preflight still carries WARNINGs/UNVERIFIABLE. CPU triage against CLAUDE_SIDECAR_ABI_ADJUDICATION.",
            required_lanes=(LANE_CPU, LANE_ANALYSIS),
            gain=INFO_HIGH,
            species="independent_reproduction",
            verifier="future.static_kernel_verify.warning_triage",
            evidence=(f"{ev}/STATIC_KERNEL_PREFLIGHT.json", f"{ev}/CLAUDE_SIDECAR_ABI_ADJUDICATION.json", f"{ev}/ABI_VERDICT_HARNESS.json"),
            hypothesis_family="static_kernel_warning_triage",
            source_f="F018",
        ),
        _item(
            id="FT.GPU_KERNELS.ready-protected",
            frontier="GPU_KERNELS",
            kind="BLOCKED",
            title="READY_PROTECTED Qwen27 candidates wait on a GPU window this sidecar must not seize",
            detail="Identity set is filled from the live queue (or recovered snapshot) at load. One SLEEPING unit, not one unit per candidate.",
            required_lanes=(LANE_GPU_PROTECTED,),
            gain=INFO_HIGH,
            species="accelerator_candidate_qualification",
            verifier="accelerator.physical.ready_protected_batch",
            evidence=(QUEUE_REL, f"{ev}/CANDIDATE_STAGED_PLAN.json", f"{ev}/QUALIFICATION_PIPELINE.json"),
            hypothesis_family="ready_protected_qualification",
            resource_class="GPU_EXCLUSIVE",
            wake_all_of=_WAKE_GPU,
            source_f="F002",
        ),
        _item(
            id="FT.GPU_KERNELS.flash-nx",
            frontier="GPU_KERNELS",
            kind="BLOCKED",
            title="Flash source-independent NX remains SCAFFOLD_ONLY / sealed-metadata",
            detail="Dominant blocker: 12+ Flash candidates collapse to this missing dependency. Audit is CPU; qualification is GPU. This item is the qualification half and sleeps.",
            required_lanes=(LANE_GPU_PROTECTED,),
            gain=INFO_HIGH,
            species="accelerator_candidate_qualification",
            verifier="accelerator.physical.flash_nx",
            evidence=(f"{ev}/FLASH_NX_COMPLETENESS_AUDIT.json",),
            hypothesis_family="flash_source_independent_nx",
            resource_class="GPU_EXCLUSIVE",
            wake_all_of=_WAKE_GPU + _WAKE_NX,
            source_f="F001",
        ),
        _item(
            id="FT.STATE.coverage-audit",
            frontier="STATE",
            kind="NEXT_WORK",
            title="Audit which state organs are actually in the candidate graph vs merely named",
            detail="Flash critical-path blockers: attention and recurrent state organs are not in the candidate graph. CPU coverage audit against stateful receipts.",
            required_lanes=(LANE_CPU, LANE_ANALYSIS),
            gain=INFO_HIGH,
            species="independent_reproduction",
            verifier="future.state.organ_coverage",
            evidence=(f"{ev}/FLASH_NX_COMPLETENESS_AUDIT.json", f"{ev}/PHYSICAL_PRIMITIVES.json"),
            hypothesis_family="state_organ_coverage",
            source_f="F001",
        ),
        _item(
            id="FT.STATE.full-kv",
            frontier="STATE",
            kind="BLOCKED",
            title="Complete 48-layer per-layer KV integration is absent",
            detail="Newest stateful attention receipt proves two-position persistent KV slots; full 48-layer executor integration remains open. Sleeps.",
            required_lanes=(LANE_GPU_PROTECTED,),
            gain=INFO_HIGH,
            species="accelerator_candidate_qualification",
            verifier="accelerator.physical.full_kv",
            evidence=(HANDOFF_REL,),
            hypothesis_family="full_layer_kv",
            resource_class="GPU_EXCLUSIVE",
            wake_all_of=_WAKE_GPU,
        ),
        _item(
            id="FT.DECODING.cost-model",
            frontier="DECODING",
            kind="NEXT_WORK",
            title="Refresh decode cost models (accepted-token, KV byte classes); capability stays ABSENT until gated",
            detail="decode_civilization.py already refuses to treat byte savings as a GO. Next unit is a refresh against current organ maps.",
            required_lanes=(LANE_CPU, LANE_ANALYSIS),
            gain=INFO_HIGH,
            species="independent_reproduction",
            verifier="future.decode_civilization.cost_models",
            evidence=(f"{ev}/DECODE_CIVILIZATION.json",),
            hypothesis_family="decode_cost_models",
        ),
        _item(
            id="FT.DECODING.speculative",
            frontier="DECODING",
            kind="OPEN_QUESTION",
            title="Which speculative-decode interfaces are wired vs still a named hole?",
            detail="decode_civilization.speculative_interfaces is the inventory. CPU question; no TPS claim.",
            required_lanes=(LANE_CPU, LANE_ANALYSIS),
            gain=INFO_MEDIUM,
            species="independent_reproduction",
            verifier="future.decode_civilization.speculative_interfaces",
            evidence=(f"{ev}/DECODE_CIVILIZATION.json",),
            hypothesis_family="speculative_decode_interfaces",
        ),
        _item(
            id="FT.HCLI_SELF.emit-workunits",
            frontier="HCLI_SELF",
            kind="NEXT_WORK",
            title="Emit this book's next_work as HCLI WorkUnits the resident can schedule",
            detail="A subsystem is not operational until the resident can discover, invoke, schedule and verify it. This unit is that wiring, via the landed species constructor.",
            required_lanes=(LANE_CPU, LANE_TOOLING),
            gain=INFO_HIGH,
            species="independent_reproduction",
            verifier="future.frontiers.resident_callable",
            evidence=(f"{ev}/HCLI_FUTURE_WORKUNITS.json", f"{ev}/RESIDENT_OPTIMIZER.json"),
            hypothesis_family="hcli_frontier_callable",
        ),
        _item(
            id="FT.HCLI_SELF.no-launch",
            frontier="HCLI_SELF",
            kind="BLOCKED",
            title="A live resident model process must not be started from this sidecar",
            detail="Contract forbids starting a resident model process or taking a GPU lease. The launch half sleeps on an authorized HCLI resident lane, which this campaign is not.",
            required_lanes=(LANE_GPU_PROTECTED,),
            gain=INFO_MEDIUM,
            species="independent_reproduction",
            verifier="future.resident_install.no_launch",
            evidence=(f"{ev}/RESIDENT_INSTALL_CONTRACT.json",),
            hypothesis_family="resident_process_launch",
            resource_class="GPU_EXCLUSIVE",
            wake_all_of=("an authorized HCLI resident-install lane distinct from this sidecar",),
        ),
        _item(
            id="FT.EXPERIMENT_TURNAROUND.refresh",
            frontier="EXPERIMENT_TURNAROUND",
            kind="NEXT_WORK",
            title="Refresh CPU experiment-turnaround receipt; cargo/GPU phases remain UNKNOWN",
            detail="Same shape as token latency, currently all-null on the Accelerator scoreboard for GPU phases. CPU phases are the honest work.",
            required_lanes=(LANE_CPU, LANE_TOOLING),
            gain=INFO_MEDIUM,
            species="independent_reproduction",
            verifier="future.turnaround.refresh",
            evidence=(f"{ev}/EXPERIMENT_TURNAROUND.json",),
            hypothesis_family="experiment_turnaround_cpu",
        ),
        _item(
            id="FT.EXPERIMENT_TURNAROUND.cargo",
            frontier="EXPERIMENT_TURNAROUND",
            kind="BLOCKED",
            title="compile/link/shader_compile/launch/execution phases require cargo and/or GPU",
            detail="This campaign must not run cargo build or touch the GPU. Those phases sleep.",
            required_lanes=(LANE_GPU_PROTECTED,),
            gain=INFO_LOW,
            species="independent_reproduction",
            verifier="future.turnaround.gpu_phases",
            evidence=(f"{ev}/EXPERIMENT_TURNAROUND.json",),
            hypothesis_family="turnaround_gpu_phases",
            resource_class="GPU_EXCLUSIVE",
            wake_all_of=_WAKE_GPU + ("cargo build authorized without contending for the shared target-dir",),
        ),
        _item(
            id="FT.PHYSICAL_GRAPH.p6-projection",
            frontier="PHYSICAL_GRAPH",
            kind="NEXT_WORK",
            title="Project Codex P6/P7 physical concepts onto Hawking primitives / HWIR / FPGA / transfer",
            detail="F020: fourteen P6/P7 candidates encode reusable concepts and have no primitive, HWIR hypothesis, FPGA realization, transfer scope or Odyssey III counterexample. All BLOCKED and unmeasured — project, do not claim.",
            required_lanes=(LANE_CPU, LANE_ANALYSIS),
            gain=INFO_HIGH,
            species="hardware_doctor_experiment",
            verifier="future.p6_projection.project",
            evidence=(f"{ev}/P6_PRIMITIVE_PROJECTION.json", f"{ev}/PHYSICAL_PRIMITIVES.json", f"{ev}/HWIR_V1.json"),
            hypothesis_family="p6_p7_projection",
            source_f="F020",
        ),
        _item(
            id="FT.PHYSICAL_GRAPH.hwir-lower",
            frontier="PHYSICAL_GRAPH",
            kind="NEXT_WORK",
            title="Lower current PhysicalGraph-shaped receipts into HWIR node/edge records",
            detail="hwir.py exists (F003 closed by sidecar). Remaining work is consumption, not a second IR.",
            required_lanes=(LANE_CPU, LANE_ANALYSIS),
            gain=INFO_MEDIUM,
            species="hardware_doctor_experiment",
            verifier="future.hwir.lower",
            evidence=(f"{ev}/HWIR_V1.json", f"{ev}/PHYSICAL_PRIMITIVES.json"),
            hypothesis_family="hwir_lowering",
            source_f="F003",
        ),
        _item(
            id="FT.FPGA.engine-sim",
            frontier="FPGA",
            kind="NEXT_WORK",
            title="Simulate FPGA engine-school goldens on CPU; associativity is part of the number",
            detail="FPGA is Accelerator/Physical Compiler/Fusion, not its own civilization. No HDL emit. No board.",
            required_lanes=(LANE_CPU, LANE_SIMULATION),
            gain=INFO_HIGH,
            species="fpga_simulation",
            verifier="future.fpga_engines.qgemv",
            evidence=(f"{ev}/FPGA_ENGINE_SCHOOL.json", f"{ev}/FPGA_MULTIFIDELITY.json"),
            hypothesis_family="fpga_engine_simulation",
            resource_class="COMPILE",
            effect_class="REVERSIBLE",
        ),
        _item(
            id="FT.FPGA.hardware-doctor",
            frontier="FPGA",
            kind="NEXT_WORK",
            title="Rank Hardware Doctor experiments against atlas hypotheses; ranking is not execution",
            detail="F004 closed by sidecar. Next: keep the ranked queue current as Codex adds P6/P7 concepts.",
            required_lanes=(LANE_CPU, LANE_ANALYSIS),
            gain=INFO_MEDIUM,
            species="hardware_doctor_experiment",
            verifier="future.hardware_doctor.rank",
            evidence=(f"{ev}/HARDWARE_DOCTOR.json",),
            hypothesis_family="hardware_doctor_rank",
            source_f="F004",
        ),
        _item(
            id="FT.FPGA.u50",
            frontier="FPGA",
            kind="BLOCKED",
            title="U50 board arrival work sleeps until the board is here",
            detail="No U50. device_ascension_pipeline.py is the arrival floor; it does not impersonate a board.",
            required_lanes=(LANE_FPGA,),
            gain=INFO_HIGH,
            species="fpga_simulation",
            verifier="future.device_ascension.u50",
            evidence=(f"{ev}/DEVICE_ASCENSION_PIPELINE.json",),
            hypothesis_family="u50_arrival",
            resource_class="COMPILE",
            wake_all_of=_WAKE_FPGA,
        ),
        _item(
            id="FT.ANE.preboard",
            frontier="ANE",
            kind="NEXT_WORK",
            title="Keep the ANE preboard graph corpus current; execution entry points must keep raising while the toolchain is missing",
            detail="ane_preboard.py is STATIC_ONLY. estimate_ane_latency / measure_prediction must raise. CPU work is corpus + placement schema, not an ANE number.",
            required_lanes=(LANE_CPU, LANE_ANALYSIS),
            gain=INFO_MEDIUM,
            species="independent_reproduction",
            verifier="future.ane_preboard.corpus",
            evidence=(f"{ev}/ANE_PREBOARD.json",),
            hypothesis_family="ane_preboard_corpus",
        ),
        _item(
            id="FT.ANE.execution",
            frontier="ANE",
            kind="BLOCKED",
            title="Flash-shaped ANE execution / latency / energy / residency is not authorized",
            detail="APPLE_ANE_ATLAS is ATLAS_SCAFFOLD_COMPILE_BOUNDARY; device profile PLAN_READY. No Flash-shaped ANE result. Sleeps.",
            required_lanes=(LANE_ANE,),
            gain=INFO_HIGH,
            species="architecture_transfer",
            verifier="accelerator.ane.flash_shaped",
            evidence=(HANDOFF_REL, f"{hd}/APPLE_ANE_ATLAS.json"),
            hypothesis_family="ane_flash_execution",
            resource_class="GPU_EXCLUSIVE",
            wake_all_of=_WAKE_ANE,
        ),
        _item(
            id="FT.ARCHITECTURE_REPATRIATION.compile-specs",
            frontier="ARCHITECTURE_REPATRIATION",
            kind="NEXT_WORK",
            title="Compile repatriation experiment specs as STATIC_ONLY; do not run them",
            detail="architecture_transfer species copies live work_unit fields. CPU compile of specs from the repatriation queue / atlas.",
            required_lanes=(LANE_CPU, LANE_ANALYSIS),
            gain=INFO_MEDIUM,
            species="architecture_transfer",
            verifier="future.architecture_transfer.compile_spec",
            evidence=(f"{ev}/CANDIDATE_STAGED_PLAN.json", f"{ev}/PHYSICAL_PRIMITIVES.json"),
            hypothesis_family="repatriation_spec_compile",
            source_f="F012",
        ),
        _item(
            id="FT.ARCHITECTURE_REPATRIATION.device-run",
            frontier="ARCHITECTURE_REPATRIATION",
            kind="BLOCKED",
            title="Metal/CUDA/ANE repatriation runs wait on their devices",
            detail="CUDA is a source school on this Apple host, not an execution backend. Metal window is blocked. ANE execution is blocked. Sleeps.",
            required_lanes=(LANE_GPU_PROTECTED,),
            gain=INFO_MEDIUM,
            species="architecture_transfer",
            verifier="accelerator.repatriation.device_run",
            evidence=(f"{ev}/CUDA_LOWBIT_HYPOTHESES.json",),
            hypothesis_family="repatriation_device_run",
            resource_class="GPU_EXCLUSIVE",
            wake_all_of=_WAKE_GPU,
        ),
        _item(
            id="FT.TOOLS.freshness",
            frontier="TOOLS",
            kind="NEXT_WORK",
            title="Resync derived artifacts with semantic fingerprints, not sha-only",
            detail="F017: Codex rewrote the queue 37→41→44→47→49. freshness.py distinguishes STALE_FINGERPRINT_ONLY vs STALE_SEMANTIC. Run it.",
            required_lanes=(LANE_CPU, LANE_TOOLING),
            gain=INFO_HIGH,
            species="independent_reproduction",
            verifier="future.freshness.resync",
            evidence=(f"{ev}/DERIVED_FRESHNESS.json", f"{ev}/CANDIDATE_STAGED_PLAN.json"),
            hypothesis_family="derived_freshness_resync",
            source_f="F017",
        ),
        _item(
            id="FT.TOOLS.propagate-skips",
            frontier="TOOLS",
            kind="NEXT_WORK",
            title="Inspect why propagate applied 0 records and skipped every delta as duplicate",
            detail="F016 closed the routing loop; PROPAGATION_STATE shows applied=0 and skipped_as_duplicate in the hundreds. That is not compounding. CPU work: name the real novel deltas, if any.",
            required_lanes=(LANE_CPU, LANE_TOOLING),
            gain=INFO_HIGH,
            species="independent_reproduction",
            verifier="future.propagate.novel_deltas",
            evidence=(f"{ev}/PROPAGATION_STATE.json", f"{ev}/CODEX_INGEST_STATE.json"),
            hypothesis_family="propagate_novel_deltas",
            source_f="F016",
        ),
        _item(
            id="FT.TOOLS.frontiers-refill",
            frontier="TOOLS",
            kind="NEXT_WORK",
            title="Rebuild this frontier book from disk after every sealed sidecar receipt",
            detail="The idle state is the failure. Refill is the loop: discover, invoke, schedule, verify, persist, refill.",
            required_lanes=(LANE_CPU, LANE_TOOLING),
            gain=INFO_MEDIUM,
            species="independent_reproduction",
            verifier="future.frontiers.refill",
            evidence=(f"{ev}/CLAUDE_GLOBAL_FRONTIER.json",),
            hypothesis_family="frontier_refill",
        ),
        _item(
            id="FT.CONTEXT.disk-authority",
            frontier="CONTEXT",
            kind="NEXT_WORK",
            title="Map HCLI context surfaces onto receipts: disk is authority, models think, context is a cache",
            detail="Any context blob that is treated as a source of truth against a sealed receipt is a bug. CPU audit of what HCLI context caches vs what receipts decide.",
            required_lanes=(LANE_CPU, LANE_ANALYSIS),
            gain=INFO_HIGH,
            species="independent_reproduction",
            verifier="future.context.disk_authority",
            evidence=(f"{ev}/EVIDENCE_SNAPSHOT.json", f"{ev}/CODEX_INGEST_STATE.json"),
            hypothesis_family="context_is_a_cache",
        ),
        _item(
            id="FT.CONTEXT.open-question",
            frontier="CONTEXT",
            kind="OPEN_QUESTION",
            title="Which HCLI context keys are allowed to outlive the receipt that minted them?",
            detail="A cache without a TTL against disk state is how stale context becomes a competing authority.",
            required_lanes=(LANE_CPU, LANE_ANALYSIS),
            gain=INFO_MEDIUM,
            species="independent_reproduction",
            verifier="future.context.ttl",
            evidence=(f"{ev}/EVIDENCE_SNAPSHOT.json",),
            hypothesis_family="context_ttl",
        ),
        _item(
            id="FT.MEMORY.hmf",
            frontier="MEMORY",
            kind="NEXT_WORK",
            title="Audit HMF managed-object legal transitions; tri-state coherence must not collapse to a boolean",
            detail="HMF_MANAGED_OBJECTS.json is executable. Next: walk object_legal_transitions against current organ maps.",
            required_lanes=(LANE_CPU, LANE_ANALYSIS),
            gain=INFO_MEDIUM,
            species="independent_reproduction",
            verifier="future.hmf.transitions",
            evidence=(f"{ev}/HMF_MANAGED_OBJECTS.json",),
            hypothesis_family="hmf_legal_transitions",
        ),
        _item(
            id="FT.MEMORY.hbm",
            frontier="MEMORY",
            kind="NEXT_WORK",
            title="Keep the HBM residency ranking current as organ census fields arrive",
            detail="Distinct from ACTIVE_BYTES ranking: this is the memory-tier identity question (MoveOrRecompute vs PersistentPhysicalRegion).",
            required_lanes=(LANE_CPU, LANE_ANALYSIS),
            gain=INFO_MEDIUM,
            species="hardware_doctor_experiment",
            verifier="future.hbm_doctor.memory_tiers",
            evidence=(f"{ev}/HBM_DOCTOR.json", f"{ev}/PHYSICAL_PRIMITIVES.json"),
            hypothesis_family="memory_tier_identity",
            source_f="F005",
        ),
        _item(
            id="FT.VERIFICATION.negative-index",
            frontier="VERIFICATION",
            kind="NEXT_WORK",
            title="Query the negative-science index before every new experiment proposal (this module's admit() is that gate)",
            detail="F009: rediscovery was free. admit() must keep refusing scar-eligible families. CPU: keep the index current via ingest, do not restatement.",
            required_lanes=(LANE_CPU, LANE_TOOLING),
            gain=INFO_HIGH,
            species="independent_reproduction",
            verifier="future.negative_index.refuse_if_dead",
            evidence=(f"{ev}/NEGATIVE_SCIENCE_INDEX.json",),
            hypothesis_family="negative_index_admission_gate",
            source_f="F009",
        ),
        _item(
            id="FT.VERIFICATION.repro",
            frontier="VERIFICATION",
            kind="NEXT_WORK",
            title="Build an independent reproduction bundle with fault injection for one sealed Codex receipt",
            detail="F008: autonomy can launder weak evidence. repro_science.py exists. Replica does not become the source.",
            required_lanes=(LANE_CPU, LANE_ANALYSIS),
            gain=INFO_HIGH,
            species="independent_reproduction",
            verifier="future.repro.bundle",
            evidence=(f"{ev}/REPRO_SCIENCE.json",),
            hypothesis_family="replication_bundle",
            resource_class="TEST",
            source_f="F008",
        ),
        _item(
            id="FT.VERIFICATION.contamination",
            frontier="VERIFICATION",
            kind="OPEN_QUESTION",
            title="Is contamination metadata actually a promotion gate yet, or still a record?",
            detail="F011. contamination.py exists. A DIAGNOSTIC_RELATIVE number must not promote as PROTECTED_ABSOLUTE. This sidecar produces neither; it only refuses the promotion path.",
            required_lanes=(LANE_CPU, LANE_ANALYSIS),
            gain=INFO_MEDIUM,
            species="independent_reproduction",
            verifier="future.contamination.promotion_gate",
            evidence=(f"{ev}/CONTAMINATION_SCIENCE.json",),
            hypothesis_family="contamination_promotion_gate",
            source_f="F011",
        ),
        _item(
            id="FT.ODYSSEY_TRANSFER.re-earn",
            frontier="ODYSSEY_TRANSFER",
            kind="NEXT_WORK",
            title="Propose re-earn of MODEL_LOCAL Odyssey II laws on named targets without widening scope",
            detail="odyssey2_law_store.py holds  the seeded laws. GENERIC_VERIFIED is empty. Transfer is a PROPOSAL, not Law.promote().",
            required_lanes=(LANE_CPU, LANE_ODYSSEY),
            gain=INFO_HIGH,
            species="odyssey_ii_transfer_experiment",
            verifier="future.odyssey_ii.law_scope",
            evidence=(f"{ev}/ODYSSEY2_LAW_STORE.json",),
            hypothesis_family="odyssey2_reearn_model_local",
            source_f="F010",
        ),
        _item(
            id="FT.ODYSSEY_TRANSFER.flash-qwen27",
            frontier="ODYSSEY_TRANSFER",
            kind="OPEN_QUESTION",
            title="Which Flash↔Qwen27 transfer hypotheses are still unevidenced at ARCHITECTURE_FAMILY?",
            detail="The transfer school exists; GENERIC_VERIFIED is 0. CPU: list laws whose scope exceeds evidence_strength.",
            required_lanes=(LANE_CPU, LANE_ODYSSEY),
            gain=INFO_MEDIUM,
            species="odyssey_ii_transfer_experiment",
            verifier="future.odyssey_ii.scope_vs_evidence",
            evidence=(f"{ev}/ODYSSEY2_LAW_STORE.json",),
            hypothesis_family="flash_qwen27_transfer_scope",
        ),
        _item(
            id="FT.ODYSSEY_ADVERSARY.attacks",
            frontier="ODYSSEY_ADVERSARY",
            kind="NEXT_WORK",
            title="Generate Odyssey III attack specs against every current law; a law with no attack is refused",
            detail="odyssey3_adversary.py: LAW -> TRANSFER HYPOTHESIS -> ADVERSARIAL TARGET -> EXPERIMENT SPEC. STATIC_ONLY specs, not measurements.",
            required_lanes=(LANE_CPU, LANE_ODYSSEY),
            gain=INFO_HIGH,
            species="odyssey_iii_adversarial_experiment",
            verifier="future.odyssey_iii.adversary",
            evidence=(f"{ev}/ODYSSEY3_ADVERSARY.json", f"{ev}/ODYSSEY2_LAW_STORE.json"),
            hypothesis_family="odyssey3_attack_generation",
        ),
        _item(
            id="FT.ODYSSEY_ADVERSARY.closed-loop",
            frontier="ODYSSEY_ADVERSARY",
            kind="OPEN_QUESTION",
            title="Has any refutation actually moved a law scope DOWN, or is the loop still open?",
            detail="A refutation that does not move scope DOWN is a bug in the loop. Disk is authority.",
            required_lanes=(LANE_CPU, LANE_ODYSSEY),
            gain=INFO_HIGH,
            species="odyssey_iii_adversarial_experiment",
            verifier="future.odyssey_iii.scope_moved",
            evidence=(f"{ev}/ODYSSEY3_ADVERSARY.json",),
            hypothesis_family="odyssey3_scope_moved",
        ),
        _item(
            id="FT.CHILD_RESIDENT.install-dry-run",
            frontier="CHILD_RESIDENT",
            kind="NEXT_WORK",
            title="Dry-run the resident install contract against Flash NX / Qwen genomes without launching a process",
            detail="resident_install.py bind_winner / validate_contract. Must not launch, install, or unload a resident.",
            required_lanes=(LANE_CPU, LANE_TOOLING),
            gain=INFO_HIGH,
            species="independent_reproduction",
            verifier="future.resident_install.dry_run",
            evidence=(f"{ev}/RESIDENT_INSTALL_CONTRACT.json", f"{ev}/RESIDENT_OPTIMIZER.json"),
            hypothesis_family="resident_install_dry_run",
            source_f="F013",
        ),
        _item(
            id="FT.CHILD_RESIDENT.optimizer",
            frontier="CHILD_RESIDENT",
            kind="NEXT_WORK",
            title="Emit bounded resident-optimizer hypotheses; promote() must keep not existing",
            detail="resident_optimizer.py is BUILT_NOT_PROMOTED. Next: generate() against current LPC / law-store / hardware-doctor receipts.",
            required_lanes=(LANE_CPU, LANE_ANALYSIS),
            gain=INFO_MEDIUM,
            species="learned_compiler_experiment",
            verifier="future.resident_optimizer.generate",
            evidence=(f"{ev}/RESIDENT_OPTIMIZER.json",),
            hypothesis_family="resident_optimizer_hypotheses",
        ),
        _item(
            id="FT.CHILD_RESIDENT.launch",
            frontier="CHILD_RESIDENT",
            kind="BLOCKED",
            title="Launching a child resident / taking a GPU lease is forbidden in this sidecar",
            detail="Sleeps on an authorized HCLI resident lane. Never a synthetic 'installed' result.",
            required_lanes=(LANE_GPU_PROTECTED,),
            gain=INFO_LOW,
            species="independent_reproduction",
            verifier="future.resident_install.launch",
            evidence=(f"{ev}/RESIDENT_INSTALL_CONTRACT.json",),
            hypothesis_family="child_resident_launch",
            resource_class="GPU_EXCLUSIVE",
            wake_all_of=("authorized HCLI resident-install lane", "existing HCLI lease with proven holder"),
        ),
        _item(
            id="FT.MODEL_EXECUTION.capacity-occupancy",
            frontier="MODEL_EXECUTION",
            kind="NEXT_WORK",
            title="Capacity class B: occupancy",
            detail=(
                "MULTISTREAM_CAPACITY_TREE class B, still OPEN. G009 measured "
                "~361 GB/s at n=1 against 449-580 aggregate, so the machine has "
                "capacity one stream does not use; SINGLE_TOKEN_PARALLEL_SLACK "
                "killed the artificial-serialization explanation (11 edges, 11 true "
                "dependencies) and INTRA_TOKEN_CONCURRENCY_AB confirmed it from the "
                "other side (0.62 microseconds between arms). What remains is inside "
                "the kernels. Discriminator: sweep threadgroup size and grid on ONE production matvec at fixed bytes and fixed arithmetic; if effective GB/s rises with occupancy alone the class is live."
            ),
            required_lanes=(LANE_GPU_PROTECTED,),
            wake_all_of=_WAKE_GPU,
            wake_never=_WAKE_NEVER,
            gain=INFO_HIGH,
            species="executor_capacity_discriminator",
            verifier="future.multistream_capacity_tree.classes",
            evidence=(f"{ev}/MULTISTREAM_CAPACITY_TREE.json",
                      f"{ev}/RESIDENT_CONCURRENCY_MEASURED.json",
                      f"{ev}/SINGLE_TOKEN_PARALLEL_SLACK.json"),
            hypothesis_family="executor_capacity_occupancy",
            source_f="S025",
        ),
        _item(
            id="FT.MODEL_EXECUTION.capacity-memory_level_parallelism",
            frontier="MODEL_EXECUTION",
            kind="NEXT_WORK",
            title="Capacity class E: memory level parallelism",
            detail=(
                "MULTISTREAM_CAPACITY_TREE class E, still OPEN. G009 measured "
                "~361 GB/s at n=1 against 449-580 aggregate, so the machine has "
                "capacity one stream does not use; SINGLE_TOKEN_PARALLEL_SLACK "
                "killed the artificial-serialization explanation (11 edges, 11 true "
                "dependencies) and INTRA_TOKEN_CONCURRENCY_AB confirmed it from the "
                "other side (0.62 microseconds between arms). What remains is inside "
                "the kernels. Discriminator: unroll the per-thread load stride on one matvec so each thread issues N independent loads before consuming any; bytes and arithmetic identical, output bit-identical."
            ),
            required_lanes=(LANE_GPU_PROTECTED,),
            wake_all_of=_WAKE_GPU,
            wake_never=_WAKE_NEVER,
            gain=INFO_HIGH,
            species="executor_capacity_discriminator",
            verifier="future.multistream_capacity_tree.classes",
            evidence=(f"{ev}/MULTISTREAM_CAPACITY_TREE.json",
                      f"{ev}/RESIDENT_CONCURRENCY_MEASURED.json",
                      f"{ev}/SINGLE_TOKEN_PARALLEL_SLACK.json"),
            hypothesis_family="executor_capacity_memory_level_parallelism",
            source_f="S025",
        ),
        _item(
            id="FT.MODEL_EXECUTION.capacity-instruction_dependency_chain",
            frontier="MODEL_EXECUTION",
            kind="NEXT_WORK",
            title="Capacity class D: instruction dependency chain",
            detail=(
                "MULTISTREAM_CAPACITY_TREE class D, still OPEN. G009 measured "
                "~361 GB/s at n=1 against 449-580 aggregate, so the machine has "
                "capacity one stream does not use; SINGLE_TOKEN_PARALLEL_SLACK "
                "killed the artificial-serialization explanation (11 edges, 11 true "
                "dependencies) and INTRA_TOKEN_CONCURRENCY_AB confirmed it from the "
                "other side (0.62 microseconds between arms). What remains is inside "
                "the kernels. Discriminator: the matched ARM A pair: bytes identical, arithmetic stripped. If stripping arithmetic moves the time, the chain is arithmetic-bound rather than memory-bound."
            ),
            required_lanes=(LANE_GPU_PROTECTED,),
            wake_all_of=_WAKE_GPU,
            wake_never=_WAKE_NEVER,
            gain=INFO_HIGH,
            species="executor_capacity_discriminator",
            verifier="future.multistream_capacity_tree.classes",
            evidence=(f"{ev}/MULTISTREAM_CAPACITY_TREE.json",
                      f"{ev}/RESIDENT_CONCURRENCY_MEASURED.json",
                      f"{ev}/SINGLE_TOKEN_PARALLEL_SLACK.json"),
            hypothesis_family="executor_capacity_instruction_dependency_chain",
            source_f="S025",
        ),
        _item(
            id="FT.MODEL_EXECUTION.capacity-register_limited_occupancy",
            frontier="MODEL_EXECUTION",
            kind="NEXT_WORK",
            title="Capacity class C: register limited occupancy",
            detail=(
                "MULTISTREAM_CAPACITY_TREE class C, still OPEN. G009 measured "
                "~361 GB/s at n=1 against 449-580 aggregate, so the machine has "
                "capacity one stream does not use; SINGLE_TOKEN_PARALLEL_SLACK "
                "killed the artificial-serialization explanation (11 edges, 11 true "
                "dependencies) and INTRA_TOKEN_CONCURRENCY_AB confirmed it from the "
                "other side (0.62 microseconds between arms). What remains is inside "
                "the kernels. Discriminator: read the compiled pipeline's register footprint and max threads per threadgroup from the Metal reflection and compare against the occupancy sweep."
            ),
            required_lanes=(LANE_CPU, LANE_ANALYSIS),
            gain=INFO_HIGH,
            species="executor_capacity_discriminator",
            verifier="future.multistream_capacity_tree.classes",
            evidence=(f"{ev}/MULTISTREAM_CAPACITY_TREE.json",
                      f"{ev}/RESIDENT_CONCURRENCY_MEASURED.json",
                      f"{ev}/SINGLE_TOKEN_PARALLEL_SLACK.json"),
            hypothesis_family="executor_capacity_register_limited_occupancy",
            source_f="S025",
        ),
        _item(
            id="FT.MODEL_EXECUTION.capacity-cache_behaviour",
            frontier="MODEL_EXECUTION",
            kind="NEXT_WORK",
            title="Capacity class F: cache behaviour",
            detail=(
                "MULTISTREAM_CAPACITY_TREE class F, still OPEN. G009 measured "
                "~361 GB/s at n=1 against 449-580 aggregate, so the machine has "
                "capacity one stream does not use; SINGLE_TOKEN_PARALLEL_SLACK "
                "killed the artificial-serialization explanation (11 edges, 11 true "
                "dependencies) and INTRA_TOKEN_CONCURRENCY_AB confirmed it from the "
                "other side (0.62 microseconds between arms). What remains is inside "
                "the kernels. Discriminator: vary only the per-launch working-set size at constant bytes and constant arithmetic and look for a knee."
            ),
            required_lanes=(LANE_GPU_PROTECTED,),
            wake_all_of=_WAKE_GPU,
            wake_never=_WAKE_NEVER,
            gain=INFO_HIGH,
            species="executor_capacity_discriminator",
            verifier="future.multistream_capacity_tree.classes",
            evidence=(f"{ev}/MULTISTREAM_CAPACITY_TREE.json",
                      f"{ev}/RESIDENT_CONCURRENCY_MEASURED.json",
                      f"{ev}/SINGLE_TOKEN_PARALLEL_SLACK.json"),
            hypothesis_family="executor_capacity_cache_behaviour",
            source_f="S025",
        ),
        _item(
            id="FT.MODEL_EXECUTION.capacity-command_queue_topology",
            frontier="MODEL_EXECUTION",
            kind="NEXT_WORK",
            title="Capacity class G: command queue topology",
            detail=(
                "MULTISTREAM_CAPACITY_TREE class G, still OPEN. G009 measured "
                "~361 GB/s at n=1 against 449-580 aggregate, so the machine has "
                "capacity one stream does not use; SINGLE_TOKEN_PARALLEL_SLACK "
                "killed the artificial-serialization explanation (11 edges, 11 true "
                "dependencies) and INTRA_TOKEN_CONCURRENCY_AB confirmed it from the "
                "other side (0.62 microseconds between arms). What remains is inside "
                "the kernels. Discriminator: the same token issued across two command queues versus one, identical kernels and identical order, token output bit-identical. INTRA_TOKEN_CONCURRENCY_AB varied encoder topology within ONE queue and explicitly did not test this."
            ),
            required_lanes=(LANE_GPU_PROTECTED,),
            wake_all_of=_WAKE_GPU,
            wake_never=_WAKE_NEVER,
            gain=INFO_HIGH,
            species="executor_capacity_discriminator",
            verifier="future.multistream_capacity_tree.classes",
            evidence=(f"{ev}/MULTISTREAM_CAPACITY_TREE.json",
                      f"{ev}/RESIDENT_CONCURRENCY_MEASURED.json",
                      f"{ev}/SINGLE_TOKEN_PARALLEL_SLACK.json"),
            hypothesis_family="executor_capacity_command_queue_topology",
            source_f="S025",
        ),
        _item(
            id="FT.MODEL_EXECUTION.capacity-kernel_shape_underfill",
            frontier="MODEL_EXECUTION",
            kind="NEXT_WORK",
            title="Capacity class I: kernel shape underfill",
            detail=(
                "MULTISTREAM_CAPACITY_TREE class I, still OPEN. G009 measured "
                "~361 GB/s at n=1 against 449-580 aggregate, so the machine has "
                "capacity one stream does not use; SINGLE_TOKEN_PARALLEL_SLACK "
                "killed the artificial-serialization explanation (11 edges, 11 true "
                "dependencies) and INTRA_TOKEN_CONCURRENCY_AB confirmed it from the "
                "other side (0.62 microseconds between arms). What remains is inside "
                "the kernels. Discriminator: run the SAME kernel at batch>1 on synthetic input and compare GB/s per byte moved against the batch-1 rate."
            ),
            required_lanes=(LANE_GPU_PROTECTED,),
            wake_all_of=_WAKE_GPU,
            wake_never=_WAKE_NEVER,
            gain=INFO_HIGH,
            species="executor_capacity_discriminator",
            verifier="future.multistream_capacity_tree.classes",
            evidence=(f"{ev}/MULTISTREAM_CAPACITY_TREE.json",
                      f"{ev}/RESIDENT_CONCURRENCY_MEASURED.json",
                      f"{ev}/SINGLE_TOKEN_PARALLEL_SLACK.json"),
            hypothesis_family="executor_capacity_kernel_shape_underfill",
            source_f="S025",
        ),
    )


# ---------------------------------------------------------------------------
# Wake evidence from disk. Listing a lane is not qualification.
# ---------------------------------------------------------------------------

def _wake_evidence(handoff: dict[str, Any] | None, qual: dict[str, Any] | None, nx: dict[str, Any] | None) -> dict[str, Any]:
    blockers = []
    if isinstance(handoff, dict):
        raw = handoff.get("exact_physical_blockers") or []
        if isinstance(raw, list):
            blockers = [str(x) for x in raw]
    teacher = None
    if isinstance(handoff, dict):
        teacher = handoff.get("teacher_capture_state")
    nx_status = None
    if isinstance(nx, dict):
        nx_status = (
            (nx.get("nx_completeness_checker") or {}).get("status")
            if isinstance(nx.get("nx_completeness_checker"), dict)
            else nx.get("status")
        )
        # FLASH_NX audit shape varies; also look at claim_boundary_audit / existing seals.
    flash_nx = None
    if isinstance(handoff, dict):
        flash = handoff.get("current_flash_state") or {}
        if isinstance(flash, dict):
            src = flash.get("source_independent_nx") or {}
            if isinstance(src, dict):
                flash_nx = src.get("status")
    ane_claim = None
    if isinstance(handoff, dict):
        ane = handoff.get("ane_state") or {}
        if isinstance(ane, dict):
            ane_claim = ane.get("claim")
    machine_heavy = False
    if isinstance(qual, dict):
        dry = qual.get("dry_run_stop") or {}
        if isinstance(dry, dict) and dry:
            machine_heavy = True
        text = json.dumps(qual, sort_keys=True)[:4000].lower()
        if "heavy" in text or "quiescen" in text:
            machine_heavy = True

    gpu_ok = False
    ane_ok = False
    fpga_ok = False
    nx_ok = bool(flash_nx) and str(flash_nx).upper() not in {
        "SCAFFOLD_ONLY",
        "SEALED_METADATA_ONLY_NOT_FOR_PROMOTION",
        "NOT_BUILT",
    }
    teacher_rows = 0
    teacher_min = None
    if isinstance(teacher, dict):
        teacher_rows = int(teacher.get("teacher_rows_written") or 0)
        teacher_min = teacher.get("minimum_rows")
    teacher_ok = False
    if teacher_min not in (None, "", 0):
        try:
            teacher_ok = teacher_rows >= int(teacher_min)
        except (TypeError, ValueError):
            teacher_ok = False

    blob = " ".join(blockers).lower()
    if "no metal-capable gpu" in blob or "no metal-capable gpu" in blob.replace("_", " "):
        gpu_ok = False
    if "xcrun cannot locate the metal compiler" in blob:
        gpu_ok = False
    if "teacher capture is 0/256" in blob:
        teacher_ok = False
    if "scaffold_only" in blob or "source-independent nx remains scaffold" in blob:
        nx_ok = False

    return {
        "gpu_protected_qualified": gpu_ok,
        "ane_qualified": ane_ok,
        "fpga_board_present": fpga_ok,
        "flash_nx_qualified": nx_ok,
        "teacher_capture_complete": teacher_ok,
        "teacher_rows_written": teacher_rows,
        "teacher_minimum_rows": teacher_min,
        "flash_nx_status": flash_nx or nx_status or "UNKNOWN",
        "ane_claim": ane_claim,
        "machine_classified_heavy": machine_heavy,
        "exact_physical_blockers": blockers,
        "rule": "listing a lane name is not qualification; disk evidence is",
    }


def _hardware_lane_awake(lane: str, wake: Mapping[str, Any]) -> bool:
    if lane == LANE_GPU_PROTECTED:
        return bool(wake.get("gpu_protected_qualified"))
    if lane == LANE_ANE:
        return bool(wake.get("ane_qualified"))
    if lane == LANE_FPGA:
        return bool(wake.get("fpga_board_present"))
    return True


def _item_sleeping(item: Mapping[str, Any], wake: Mapping[str, Any]) -> bool:
    if item.get("kind") == "BLOCKED":
        return True
    req = [l for l in (item.get("required_lanes") or []) if l in HARDWARE_LANES]
    if not req:
        return False
    return not all(_hardware_lane_awake(l, wake) for l in req)


def _item_runnable(item: Mapping[str, Any], available: frozenset[str], wake: Mapping[str, Any]) -> bool:
    if _item_sleeping(item, wake):
        return False
    req = frozenset(item.get("required_lanes") or [])
    if req & frozenset(HARDWARE_LANES):
        if not all(_hardware_lane_awake(l, wake) for l in req if l in HARDWARE_LANES):
            return False
    return req <= available


# ---------------------------------------------------------------------------
# WorkUnit emission (landed species constructor). Local extras named for
# wakeup.py / workgraph.py / resident_api.py to swap onto later.
# ---------------------------------------------------------------------------

def _emit_unit(item: Mapping[str, Any], *, sleeping: bool, overlay: Mapping[str, Any] | None = None) -> dict[str, Any]:
    status = "blocked" if sleeping else "pending"
    classification = "SLEEPING" if sleeping else "STATIC_ONLY"
    extras: dict[str, Any] = {
        "frontier": item["frontier"],
        "required_lanes": list(item.get("required_lanes") or []),
        "species": item.get("species"),
        "expected_information_gain": item.get("expected_information_gain"),
        "redundancy_key": _redundancy_key(item),
        "hypothesis_family": item.get("hypothesis_family"),
        "evidence_parents": list(item.get("evidence") or []),
        "wake_condition": {
            "all_of": list(item.get("wake_all_of") or []),
            "never": list(item.get("wake_never") or []),
        }
        if sleeping or item.get("wake_all_of")
        else None,
        "blocked_reason": (
            "; ".join(item.get("wake_all_of") or []) or str(item.get("detail") or "blocked")
            if sleeping
            else None
        ),
        "source_f": item.get("source_f") or None,
        "candidate_id": item.get("candidate_id") or None,
        "claim_boundary": PROPOSAL_CLAIM_BOUNDARY,
        "requires_quiescence": str(item.get("resource_class") or "") == "GPU_EXCLUSIVE",
        "candidate_status": "SLEEPING" if sleeping else "STATIC_ONLY",
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "kind": item.get("kind"),
        "title": item.get("title"),
    }
    if overlay:
        extras.update({k: v for k, v in overlay.items() if v is not None})
    row = wus.emit_hcli_workunit(
        id=str(item["id"]),
        role="science",
        description=str(item.get("detail") or item.get("title") or item["id"]),
        dependencies=[],
        resource_class=str(item.get("resource_class") or "STATIC_ANALYSIS"),
        verifier=str(item.get("verifier") or "future.frontiers.verify"),
        provider="future.frontiers",
        effect_class=str(item.get("effect_class") or "READ_ONLY"),
        status=status,
        classification=classification,
        extras=extras,
    )
    wus.validate_emitted_unit(row)
    return row


# ---------------------------------------------------------------------------
# Busywork admission
# ---------------------------------------------------------------------------

def _scar_refuse(proposal: Mapping[str, Any], scar_doc: dict[str, Any] | None) -> dict[str, Any] | None:
    family = proposal.get("hypothesis_family") or proposal.get("family") or proposal.get("technique")
    if not family:
        return None
    want = ni.canon_family(str(family))
    want_model = proposal.get("model")
    scars = (scar_doc or {}).get("scars") if isinstance(scar_doc, dict) else None
    if isinstance(scars, list) and scars:
        for scar in scars:
            if not isinstance(scar, dict):
                continue
            if not scar.get("refuse_eligible"):
                continue
            if scar.get("parse_status") and scar.get("parse_status") != ni.PARSED:
                continue
            if scar.get("hypothesis_family") != want:
                continue
            if want_model:
                models = list(scar.get("models") or [scar.get("model") or ni.UNRECORDED])
                hit = ni.canon_model(str(want_model))
                if hit not in models and str(want_model) not in models:
                    continue
            return {
                "refused": True,
                "reason": (
                    "known-dead hypothesis; rediscovery is not free. "
                    "Reopen only under the cited reopen_condition."
                ),
                "scar_id": scar.get("scar_id"),
                "source_path": scar.get("source_path"),
                "hypothesis_family": scar.get("hypothesis_family"),
                "model": scar.get("model"),
                "models": scar.get("models"),
                "verdict": scar.get("verdict"),
                "reopen_condition": scar.get("reopen_condition"),
                "level": scar.get("level"),
            }
        return None
    try:
        return ni.refuse_if_dead(dict(proposal))
    except Exception as exc:  # index unavailable in this checkout: do not block all work
        return {
            "refused": False,
            "scar_index": "unavailable",
            "note": f"scar check skipped: {exc}",
        }


def _gain_of(proposal: Mapping[str, Any], book_items: Sequence[Mapping[str, Any]]) -> int:
    if proposal.get("expected_information_gain") not in (None, ""):
        try:
            g = int(proposal["expected_information_gain"])
            return max(INFO_NONE, min(INFO_HIGH, g))
        except (TypeError, ValueError):
            pass
    key = _redundancy_key(proposal)
    for item in book_items:
        if item.get("kind") == "OPEN_QUESTION" and _redundancy_key(item) == key:
            return INFO_HIGH
        fam = proposal.get("hypothesis_family")
        if fam and item.get("kind") == "OPEN_QUESTION" and item.get("hypothesis_family") == fam:
            return INFO_HIGH
    title = _tokens(str(proposal.get("title") or proposal.get("description") or ""))
    best = INFO_NONE
    for item in book_items:
        if item.get("kind") != "OPEN_QUESTION":
            continue
        score = _jaccard(title, _tokens(str(item.get("title") or "") + " " + str(item.get("detail") or "")))
        if score >= 0.45:
            best = max(best, INFO_MEDIUM)
    if best:
        return best
    if proposal.get("hypothesis_family") or proposal.get("frontier"):
        return INFO_LOW
    return INFO_NONE


def _redundancy_of(
    proposal: Mapping[str, Any], queued: Sequence[Mapping[str, Any]]
) -> tuple[float, str | None]:
    key = _redundancy_key(proposal)
    pid = str(proposal.get("id") or "")
    ptok = _tokens(
        " ".join(
            str(proposal.get(k) or "")
            for k in ("title", "description", "detail", "hypothesis_family")
        )
    )
    best = 0.0
    who: str | None = None
    for q in queued:
        qid = str(q.get("id") or "")
        if pid and qid and pid == qid:
            return 1.0, qid
        if _redundancy_key(q) == key:
            return 1.0, qid or _redundancy_key(q)
        qtok = _tokens(
            " ".join(str(q.get(k) or "") for k in ("title", "description", "detail", "hypothesis_family"))
        )
        score = _jaccard(ptok, qtok)
        if score > best:
            best = score
            who = qid or _redundancy_key(q)
    return best, who


def admit(
    proposal: Mapping[str, Any] | None,
    *,
    queued: Sequence[Mapping[str, Any]] | None = None,
    book_items: Sequence[Mapping[str, Any]] | None = None,
    scar_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Refuse busywork at admission. Autonomy is not a flood of low-value units."""
    if not isinstance(proposal, dict) or not proposal:
        return {
            "admitted": False,
            "refused": True,
            "reason": "empty proposal",
            "expected_information_gain": INFO_NONE,
            "redundancy": 1.0,
            "scar_overlap": None,
        }
    queued = list(queued or [])
    book_items = list(book_items or [])
    gain = _gain_of(proposal, book_items)
    red, rival = _redundancy_of(proposal, queued)
    scar = _scar_refuse(proposal, scar_doc)
    if isinstance(scar, dict) and scar.get("refused"):
        return {
            "admitted": False,
            "refused": True,
            "reason": scar.get("reason") or "known-dead hypothesis",
            "expected_information_gain": gain,
            "redundancy": red,
            "scar_overlap": scar,
            "rival_id": rival,
        }
    if gain <= INFO_NONE:
        return {
            "admitted": False,
            "refused": True,
            "reason": "expected information gain is zero; a unit that cannot move a frontier is busywork",
            "expected_information_gain": gain,
            "redundancy": red,
            "scar_overlap": scar,
            "rival_id": rival,
        }
    if red >= REDUNDANCY_EXACT:
        return {
            "admitted": False,
            "refused": True,
            "reason": (
                f"redundant with already-queued work {rival!r}; "
                "a flood of the same unit is refused at admission"
            ),
            "expected_information_gain": gain,
            "redundancy": red,
            "scar_overlap": scar,
            "rival_id": rival,
        }
    if gain <= INFO_LOW and red >= REDUNDANCY_LOW_GAIN:
        return {
            "admitted": False,
            "refused": True,
            "reason": (
                f"low-information near-duplicate of {rival!r} "
                f"(gain={gain}, redundancy={red:.2f}); autonomy is not busywork"
            ),
            "expected_information_gain": gain,
            "redundancy": red,
            "scar_overlap": scar,
            "rival_id": rival,
        }
    return {
        "admitted": True,
        "refused": False,
        "reason": "novel enough: non-zero information gain, below redundancy threshold, not scar-dead",
        "expected_information_gain": gain,
        "redundancy": red,
        "scar_overlap": scar,
        "rival_id": rival,
    }


# ---------------------------------------------------------------------------
# Movement. UNKNOWN until a verified non-dominated move is recorded.
# ---------------------------------------------------------------------------

def _axes_of(point: Mapping[str, Any]) -> dict[str, int]:
    """Dimensionless count axes. Never hardware fields."""
    out: dict[str, int] = {}
    for key, value in point.items():
        if key in HARDWARE_FIELDS:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            out[key] = value
    return out


def _non_dominated(before: Mapping[str, Any], after: Mapping[str, Any], *, higher_better: Sequence[str] = ()) -> bool:
    """True iff after is better on at least one axis and not worse on the others.

    Default: lower counts of open/blocked are better; higher resolved is better.
    """
    b, a = _axes_of(before), _axes_of(after)
    keys = sorted(set(b) | set(a))
    if not keys:
        return False
    better = False
    high = set(higher_better) | {"resolved", "n_resolved", "verified_moves"}
    for k in keys:
        bv, av = b.get(k, 0), a.get(k, 0)
        if av == bv:
            continue
        if k in high:
            if av < bv:
                return False
            better = True
        else:
            if av > bv:
                return False
            better = True
    return better


def movement_unknown(reason: str = "no verified non-dominated moves have been recorded") -> dict[str, Any]:
    return {
        "primary_metric": "verified_non_dominated_frontier_moves_over_wall_time",
        "state": "UNKNOWN",
        "reason": reason + "; a fabricated baseline is refused",
        "moves": [],
        "n_moves": 0,
        "supporting_rates": {
            "moves_per_wall_second": "UNKNOWN",
            "admitted_units_per_wall_second": "UNKNOWN",
            "refused_units_per_wall_second": "UNKNOWN",
        },
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
    }


# ---------------------------------------------------------------------------
# Book
# ---------------------------------------------------------------------------

class FrontierBook:
    """Persistent 22-frontier state. Disk is authority; this object is a cache."""

    def __init__(
        self,
        *,
        items: Sequence[Mapping[str, Any]],
        recovery: Mapping[str, Any],
        wake: Mapping[str, Any],
        global_frontier: Mapping[str, Any] | None,
        scar_doc: dict[str, Any] | None,
        queue_identity: Mapping[str, Any],
        moves: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self.items: list[dict[str, Any]] = [dict(x) for x in items]
        self.recovery = dict(recovery)
        self.wake = dict(wake)
        self.global_frontier = dict(global_frontier) if isinstance(global_frontier, dict) else {}
        self.scar_doc = scar_doc
        self.queue_identity = dict(queue_identity)
        self.moves: list[dict[str, Any]] = [dict(x) for x in (moves or [])]
        self._queued_ids: list[str] = []

    def frontiers(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for name in FRONTIER_NAMES:
            mine = [i for i in self.items if i.get("frontier") == name]
            questions = [i for i in mine if i["kind"] == "OPEN_QUESTION"]
            blocked = [i for i in mine if _item_sleeping(i, self.wake) or i["kind"] == "BLOCKED"]
            nxt = [
                i
                for i in mine
                if i["kind"] == "NEXT_WORK" and not _item_sleeping(i, self.wake)
            ]
            runnable_q = [
                q
                for q in questions
                if not (set(q.get("required_lanes") or []) & set(HARDWARE_LANES))
                or all(
                    _hardware_lane_awake(l, self.wake)
                    for l in (q.get("required_lanes") or [])
                    if l in HARDWARE_LANES
                )
            ]
            if nxt or runnable_q:
                status = "ACTIVE"
            elif blocked:
                status = "BLOCKED"
            else:
                status = "EXHAUSTED"
            out[name] = {
                "name": name,
                "status": status,
                "open_questions": [_public_item(q) for q in sorted(questions, key=lambda x: x["id"])],
                "blocked": [_public_item(b) for b in sorted(blocked, key=lambda x: x["id"])],
                "next_work": [_public_item(n) for n in sorted(nxt, key=lambda x: x["id"])],
                "counts": {
                    "open_questions": len(questions),
                    "blocked": len(blocked),
                    "next_work": len(nxt),
                    "runnable_questions": len(runnable_q),
                },
            }
        return out

    def is_idle(self) -> bool:
        """True only when every frontier is exhausted or blocked.

        A hardware lane being blocked does not make the book idle while any
        other frontier still has OPEN CPU / simulation / representation /
        tooling / Odyssey work. This is the §140 failure condition, mechanical.
        """
        states = self.frontiers()
        return all(row["status"] in {"EXHAUSTED", "BLOCKED"} for row in states.values())

    def idle_proof(self) -> dict[str, Any]:
        states = self.frontiers()
        active = sorted(n for n, r in states.items() if r["status"] == "ACTIVE")
        blocked = sorted(n for n, r in states.items() if r["status"] == "BLOCKED")
        exhausted = sorted(n for n, r in states.items() if r["status"] == "EXHAUSTED")
        return {
            "is_idle": self.is_idle(),
            "rule": (
                "is_idle may only be true when every frontier is EXHAUSTED or "
                "BLOCKED. A blocked GPU/ANE lane never makes the daemon idle "
                "while CPU/Odyssey/tooling/representation work remains."
            ),
            "n_frontiers": len(FRONTIER_NAMES),
            "n_active": len(active),
            "n_blocked": len(blocked),
            "n_exhausted": len(exhausted),
            "active": active,
            "blocked": blocked,
            "exhausted": exhausted,
        }

    def next_work(self, available_lanes: Iterable[str] | str | None = None) -> list[dict[str, Any]]:
        """Safe WorkUnits for the lanes that ARE available.

        GPU_PROTECTED / ANE stay SLEEPING unless disk evidence says they
        qualified. They are never filled with a synthetic result.
        """
        available = _lanes(available_lanes)
        candidates = [
            i
            for i in self.items
            if i.get("kind") == "NEXT_WORK" and _item_runnable(i, available, self.wake)
        ]
        candidates.sort(
            key=lambda i: (-int(i.get("expected_information_gain") or 0), i["id"])
        )
        admitted: list[dict[str, Any]] = []
        queued_for_gate: list[dict[str, Any]] = []
        refusals: list[dict[str, Any]] = []
        for item in candidates:
            decision = admit(
                item,
                queued=queued_for_gate,
                book_items=self.items,
                scar_doc=self.scar_doc,
            )
            if not decision.get("admitted"):
                refusals.append({"id": item["id"], "reason": decision.get("reason")})
                continue
            unit = _emit_unit(item, sleeping=False)
            unit["admission"] = {k: decision[k] for k in ("admitted", "reason", "expected_information_gain", "redundancy")}
            admitted.append(unit)
            queued_for_gate.append(item)
        self._last_refusals = refusals
        self._queued_ids = [u["id"] for u in admitted]
        return admitted

    def sleeping_units(self) -> list[dict[str, Any]]:
        rows = []
        overlay_ready = self.queue_identity.get("ready_protected_ids") or []
        for item in sorted(self.items, key=lambda x: x["id"]):
            if not (_item_sleeping(item, self.wake) or item.get("kind") == "BLOCKED"):
                continue
            extra = {}
            if item["id"] == "FT.GPU_KERNELS.ready-protected" and overlay_ready:
                extra = {
                    "ready_protected_ids": list(overlay_ready),
                    "n_ready_protected": len(overlay_ready),
                    "detail_overlay": (
                        f"{len(overlay_ready)} READY_PROTECTED identities derived "
                        "from the recovered queue; one SLEEPING unit, not a flood"
                    ),
                }
            rows.append(_emit_unit(item, sleeping=True, overlay=extra or None))
        return rows

    def record_move(
        self,
        *,
        frontier: str,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        evidence: Sequence[str],
        verified: bool,
        higher_better: Sequence[str] = (),
    ) -> dict[str, Any]:
        if frontier not in FRONTIER_NAMES:
            raise ValueError(f"unknown frontier {frontier!r}")
        if not verified:
            raise UnverifiedMoveError(f"{frontier}: move is not verified")
        if not evidence:
            raise UnverifiedMoveError(f"{frontier}: verified move requires evidence refs")
        _assert_no_hardware_claims(dict(before), "before")
        _assert_no_hardware_claims(dict(after), "after")
        if not _non_dominated(before, after, higher_better=higher_better):
            raise DominatedMoveError(
                f"{frontier}: after does not non-dominate before on count axes"
            )
        move = {
            "frontier": frontier,
            "before": dict(before),
            "after": dict(after),
            "evidence": list(evidence),
            "verified": True,
            "non_dominated": True,
            "evidence_class": "STATIC_ONLY",
            "bench_state": "UNKNOWN",
        }
        self.moves.append(move)
        return move

    def movement(self) -> dict[str, Any]:
        if not self.moves:
            return movement_unknown()
        doc = movement_unknown("moves exist; rate over wall time is still UNKNOWN (no clock in hashed content)")
        doc["state"] = "UNKNOWN"
        doc["n_moves"] = len(self.moves)
        doc["moves"] = list(self.moves)
        doc["reason"] = (
            "verified non-dominated moves are recorded as a count; "
            "rate over wall time stays UNKNOWN because hashed content may not carry a wall clock"
        )
        return doc

    def refill(
        self,
        available_lanes: Iterable[str] | str | None = None,
        *,
        exclude: Iterable[str] = (),
    ) -> list[dict[str, Any]]:
        """Work the caller does NOT already have. Never idle while ACTIVE.

        This used to be a bare alias for next_work, so every refill returned the
        identical set the caller was already holding. The 1h autonomy timeline
        shows the consequence: four refills at t13/28/70/119 each returning the
        same 25 frontier ids, and none at all in the remaining 2864 seconds. A
        replay is not a refill, and a resident that cannot tell the difference
        cannot know when its frontier is actually exhausted.

        `exclude` is what the caller already holds. An empty result now means
        genuinely nothing new, which is a fact worth acting on.
        """
        held = {str(x) for x in exclude}
        return [i for i in self.next_work(available_lanes) if str(i.get("id")) not in held]

    def to_doc(self, *, available_lanes: Iterable[str] | str | None = None) -> dict[str, Any]:
        lanes = _lanes(available_lanes)
        units = self.next_work(lanes)
        sleeping = self.sleeping_units()
        states = self.frontiers()
        proof = self.idle_proof()
        gf = self.global_frontier
        gf_entries = gf.get("entries") if isinstance(gf.get("entries"), list) else []
        return {
            "schema": SCHEMA,
            "version": 1,
            "purpose": (
                "Twenty-two persistent operating frontiers. next_work yields "
                "safe CPU/simulation/representation/tooling/Odyssey work while "
                "GPU_PROTECTED and ANE sleep. is_idle is false until every "
                "frontier is exhausted or blocked. Busywork is refused at admission. "
                "Movement stays UNKNOWN until a verified non-dominated move exists."
            ),
            "eras": list(ERAS),
            "odysseys": [
                "I WHAT IS TRUE?",
                "II WHAT DID HAWKING ALREADY LEARN?",
                "III WHERE IS HAWKING WRONG?",
            ],
            "no_era_vi": True,
            "no_odyssey_iv": True,
            "fpga_civilization": (
                "FPGA belongs to Accelerator / Physical Compiler / Fusion; "
                "it is not its own civilization"
            ),
            "measurement_class": "STATIC_ONLY",
            "bench_state": "UNKNOWN",
            "gpu_authority": False,
            "produces_diagnostic_relative": False,
            "produces_protected_absolute": False,
            "frontier_names": list(FRONTIER_NAMES),
            "n_frontiers": len(FRONTIER_NAMES),
            "lanes": {
                "all": list(ALL_LANES),
                "cpu_class": list(CPU_LANES),
                "hardware": list(HARDWARE_LANES),
                "this_host_available": list(THIS_HOST_LANES),
                "blocked_on_this_host": list(BLOCKED_ON_THIS_HOST),
                "next_work_available_lanes": sorted(lanes),
            },
            "wake_evidence": self.wake,
            "queue_identity": self.queue_identity,
            "frontiers": states,
            "next_work": units,
            "n_next_work": len(units),
            "sleeping": sleeping,
            "n_sleeping": len(sleeping),
            "is_idle": proof["is_idle"],
            "idle_proof": proof,
            "busywork": {
                "rule": (
                    "score by expected information gain, redundancy against "
                    "already-queued work, and overlap with recorded scars; "
                    "a flood of low-value units is REFUSED at admission"
                ),
                "thresholds": {
                    "info_none_refused": INFO_NONE,
                    "redundancy_exact": REDUNDANCY_EXACT,
                    "low_gain_redundancy": REDUNDANCY_LOW_GAIN,
                },
                "admission_refusals_during_next_work": list(getattr(self, "_last_refusals", [])),
            },
            "movement": self.movement(),
            "global_frontier_consumed": {
                "path": GLOBAL_FRONTIER_REL,
                "schema": gf.get("schema"),
                "n_entries": len(gf_entries),
                "resolved_entries": list(gf.get("resolved_entries") or []),
                "stale_entries": list(gf.get("stale_entries") or []),
                "note": (
                    "CLAUDE_GLOBAL_FRONTIER is the campaign gap tracker. "
                    "This module extends it as an operating layer; it does not rewrite it."
                ),
            },
            "recovered_implementation": self.recovery,
            "gaps_closed": [
                "twenty-two named persistent frontiers, each with open questions, SLEEPING blocked items (wake conditions), and next work",
                "next_work(available_lanes) yields CPU analysis / simulation / representation / tooling / Odyssey work while GPU_PROTECTED and ANE are blocked",
                "is_idle() is a conjunction over all frontiers and is false on the real current book",
                "busywork refused at admission by information gain, queued redundancy, and scar overlap",
                "READY_PROTECTED candidates collapse to one SLEEPING unit; identity set is derived from the recovered queue, not a hard-coded N",
                "verified non-dominated frontier-move metric reports UNKNOWN until a real move is recorded; no fabricated baseline",
                "blocked physical work is SLEEPING, never a synthetic result",
                "HCLI WorkUnits emitted through the landed species constructor",
            ],
            "negative_findings": [
                "MetalContext reports no Metal-capable GPU on this host (Codex handoff / contract)",
                "xcrun cannot locate the Metal compiler under CommandLineTools",
                "protected bench lock files exist; holder pids unproven; flock would be a seizure",
                "qualification pipeline classifies the machine HEAVY and will not quiesce standing workers",
                "Flash source-independent NX is SCAFFOLD_ONLY, not qualified",
                "teacher capture is 0/256; no synthetic rows",
                "this sidecar produces neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE",
                "movement rate over wall time is UNKNOWN; hashed content must not carry a wall clock",
                "global_frontier.py F007/F008 probes still hold on *learned_physical* / *replication_bundle* even though lpc_dataset.py and repro_science.py exist — the campaign tracker is protected and is not edited here",
                "propagate.py closed the routing loop but applied 0 records (skipped_as_duplicate); compounding is not yet moving stores",
            ],
            "resident_callable": {
                "can_hcli_invoke": True,
                "invoke_path_today": (
                    "CLI: python3 tools/future/frontiers.py --next-work "
                    "--lanes CPU,SIMULATION,REPRESENTATION,TOOLING,ODYSSEY,ANALYSIS"
                ),
                "entry_point": "tools.future.frontiers:next_work / is_idle / admit / build / refill",
                "workunit_emitted": (
                    "HCLI WorkUnit via tools.future.workunit_species.emit_hcli_workunit; "
                    "extras: frontier, required_lanes, wake_condition, redundancy_key, "
                    "expected_information_gain, hypothesis_family"
                ),
                "receipt_written": f"receipts/future/{RECEIPT}",
                "frontier_fed": (
                    "all twenty-two named frontiers; CLAUDE_GLOBAL_FRONTIER is consumed "
                    "as input and is not overwritten"
                ),
                "fail_closed": [
                    "HardwareClaimError on numeric hardware fields (write_receipt)",
                    "GPU_PROTECTED / ANE / FPGA work stays SLEEPING until disk evidence qualifies the lane; listing the lane name is not qualification",
                    "busywork REFUSED at admission (zero gain, redundancy, scar-dead)",
                    "movement stays UNKNOWN until record_move(verified=True, non-dominated, evidence)",
                    "is_idle is false while any frontier is ACTIVE",
                    "unseen files are recorded as a recovery path, never as 'does not exist in the project'",
                    "does not start a resident model process and does not take a GPU lease",
                ],
                "integration_swaps": {
                    "resident_api.py": "callable RPC surface over next_work/is_idle/admit",
                    "workgraph.py": "schedule the emitted WorkUnits",
                    "wakeup.py": "evaluate wake_condition against hardware qualification",
                    "super_resident.py": "the daemon that must not idle",
                    "sandbox.py": "orchestrator sandbox this would run inside",
                    "scar_scheduling.py": "we query negative_index instead",
                    "evidence_dag.py": "we cite source receipts",
                    "protected_window.py": "never taken from this sidecar",
                    "odyssey_launch.py": "Odyssey I start under resident orchestration",
                    "succession.py / resident_identity.py": "child resident identity",
                },
            },
            "head": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        }


def _public_item(item: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "id",
        "frontier",
        "kind",
        "title",
        "detail",
        "required_lanes",
        "expected_information_gain",
        "hypothesis_family",
        "redundancy_key",
        "species",
        "verifier",
        "evidence",
        "wake_all_of",
        "wake_never",
        "source_f",
        "candidate_id",
        "evidence_class",
        "bench_state",
    )
    return {k: item.get(k) for k in keys if item.get(k) not in (None, "", [], {})}


def _queue_identity(queue: dict[str, Any] | None, handoff: dict[str, Any] | None) -> dict[str, Any]:
    ready: list[str] = []
    blocked: list[str] = []
    static_only: list[str] = []
    origin = "none"
    if isinstance(queue, dict):
        origin = "queue"
        for cand in queue.get("candidates") or []:
            if not isinstance(cand, dict):
                continue
            cid = str(cand.get("candidate_id") or "")
            st = str(cand.get("status") or "")
            if not cid:
                continue
            if st == "READY_PROTECTED":
                ready.append(cid)
            elif st == "BLOCKED":
                blocked.append(cid)
            elif st == "STATIC_ONLY":
                static_only.append(cid)
        wu = queue.get("work_units") or []
        n_wu = len(wu) if isinstance(wu, list) else 0
    else:
        n_wu = 0
    if not ready and isinstance(handoff, dict):
        cq = handoff.get("current_queue") or {}
        if isinstance(cq, dict):
            origin = origin if origin == "queue" else "handoff.current_queue"
            ready = [str(x) for x in (cq.get("ready_candidate_ids") or [])]
            blocked = [str(x) for x in (cq.get("blocked_candidate_ids") or [])]
            static_only = [str(x) for x in (cq.get("static_only_candidate_ids") or [])]
    if not ready:
        ready = list(wus.CORE_READY_QWEN27)
        origin = origin if origin != "none" else "workunit_species.CORE_READY_QWEN27"
    if not blocked:
        blocked = list(wus.CORE_BLOCKED_FLASH)
        if origin.startswith("workunit"):
            origin = "workunit_species.CORE_*"
    return {
        "origin": origin,
        "ready_protected_ids": sorted(set(ready)),
        "blocked_ids": sorted(set(blocked)),
        "static_only_ids": sorted(set(static_only)),
        "n_ready_protected": len(set(ready)),
        "n_blocked": len(set(blocked)),
        "n_static_only": len(set(static_only)),
        "n_work_units_in_queue": n_wu,
        "note": "counts derived from recovered identities; not a pinned integer",
    }


def _overlay_global_frontier(
    items: list[dict[str, Any]], gf: dict[str, Any] | None
) -> list[dict[str, Any]]:
    if not isinstance(gf, dict):
        return items
    resolved = set(gf.get("resolved_entries") or [])
    seen_ids = {i["id"] for i in items}
    covered = {
        (i.get("source_f"), i.get("frontier"))
        for i in items
        if i.get("source_f")
    }
    extras: list[dict[str, Any]] = []
    for entry in gf.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        fid = str(entry.get("id") or "")
        if not fid or fid in resolved:
            continue
        targets = F_TO_FRONTIERS.get(fid) or ()
        classification = str(entry.get("classification") or "")
        kind = "OPEN_QUESTION"
        lanes: tuple[str, ...] = (LANE_CPU, LANE_ANALYSIS)
        wake: tuple[str, ...] = ()
        resource = "STATIC_ANALYSIS"
        if classification == "BLOCKED":
            need = str(entry.get("resource_need") or "").lower()
            if "gpu" in need or "ane" in need:
                lanes = (LANE_ANE,) if ("ane" in need and "gpu" not in need) else (LANE_GPU_PROTECTED,)
                wake = _WAKE_ANE if lanes == (LANE_ANE,) else _WAKE_GPU
                resource = "GPU_EXCLUSIVE"
                kind = "BLOCKED"
            else:
                # Blocked on a person/verdict, not on hardware: still a CPU question.
                kind = "OPEN_QUESTION"
                wake = (str(entry.get("prerequisite") or "external verdict"),)
        for frontier in targets:
            if (fid, frontier) in covered:
                continue
            iid = f"FT.{frontier}.campaign.{fid}"
            if iid in seen_ids:
                continue
            extras.append(
                _item(
                    id=iid,
                    frontier=frontier,
                    kind=kind,
                    title=str(entry.get("title") or fid),
                    detail=str(entry.get("detail") or entry.get("expected_value") or fid),
                    required_lanes=lanes,
                    gain=INFO_MEDIUM if classification != "BLOCKED" else INFO_HIGH,
                    species="independent_reproduction",
                    verifier=f"future.global_frontier.{fid}",
                    evidence=(GLOBAL_FRONTIER_REL, str(entry.get("integration_target") or "")),
                    hypothesis_family=f"campaign_{fid.lower()}",
                    resource_class=resource,
                    wake_all_of=wake,
                    source_f=fid,
                )
            )
            seen_ids.add(iid)
    return items + extras


def load_book(*, overrides: Mapping[str, dict[str, Any] | None] | None = None) -> FrontierBook:
    """Recover the operating frontier from disk. overrides inject fixtures; None means unseen."""
    ov = dict(overrides or {})

    def _take(rel: str, loader=load_optional) -> tuple[dict[str, Any] | None, str]:
        if rel in ov:
            doc = ov[rel]
            return doc, ("override:None" if doc is None else "override")
        return loader(rel)

    gf, gf_origin = _take(GLOBAL_FRONTIER_REL)
    handoff, handoff_origin = _take(HANDOFF_REL)
    queue, queue_origin = _take(QUEUE_REL)
    if queue is None and QUEUE_REL not in ov:
        try:
            from tools.future.candidate_planner import QueueNotFoundError, load_queue

            try:
                q = load_queue()
                queue, queue_origin = q, str(q.get("_loaded_from") or "candidate_planner.load_queue")
            except QueueNotFoundError:
                pass
        except ImportError:
            pass
    nx, nx_origin = _take("receipts/future/FLASH_NX_COMPLETENESS_AUDIT.json")
    qual, qual_origin = _take("receipts/future/QUALIFICATION_PIPELINE.json")
    scars, scar_origin = _take(SCAR_INDEX_REL)
    o2, o2_origin = _take("receipts/future/ODYSSEY2_LAW_STORE.json")
    o3, o3_origin = _take("receipts/future/ODYSSEY3_ADVERSARY.json")
    prop, prop_origin = _take("receipts/future/PROPAGATION_STATE.json")

    wake = _wake_evidence(handoff, qual, nx)
    identity = _queue_identity(queue, handoff)
    items = [dict(x) for x in _catalog()]
    # Fill READY_PROTECTED identity into the sleeping GPU unit from disk.
    for item in items:
        if item["id"] == "FT.GPU_KERNELS.ready-protected":
            ids = identity["ready_protected_ids"]
            item["detail"] = (
                f"{len(ids)} READY_PROTECTED identities derived from {identity['origin']}: "
                + ", ".join(ids)
                + ". One SLEEPING unit; HCLI wakes it when GPU_PROTECTED qualifies. "
                "Never a synthetic result."
            )
            item["candidate_id"] = ""  # the set, not one child
    items = _overlay_global_frontier(items, gf)

    n_laws = None
    if isinstance(o2, dict):
        n_laws = (o2.get("counts") or {}).get("n_laws") or o2.get("accounting", {}).get("n_laws")
    n_attacks = None
    if isinstance(o3, dict):
        n_attacks = o3.get("n_attacks")

    recovery = {
        "already_existed": {
            "tools/future/global_frontier.py": (
                "adequate as the campaign gap tracker (F001–F020, probe-backed, "
                "RESOLVED/holding). Not adequate as the 22 named operating frontiers, "
                "next_work(available_lanes), is_idle, busywork admission, or movement metric."
            ),
            "tools/future/workunit_species.py": "HCLI WorkUnit field set and species catalog; consumed, not forked",
            "tools/future/negative_index.py": "scar refusal path; consumed at admission",
            "tools/future/candidate_planner.py": "queue recovery across worktrees; consumed if importable",
        },
        "gap_this_module_closes": (
            "the operating layer: 22 persistent frontiers the resident can "
            "discover, invoke, schedule, verify; result changes a frontier; "
            "persists; next work refills. A blocked hardware lane does not idle the daemon."
        ),
        "paths_taken": {
            GLOBAL_FRONTIER_REL: gf_origin,
            HANDOFF_REL: handoff_origin,
            QUEUE_REL: queue_origin,
            "receipts/future/FLASH_NX_COMPLETENESS_AUDIT.json": nx_origin,
            "receipts/future/QUALIFICATION_PIPELINE.json": qual_origin,
            SCAR_INDEX_REL: scar_origin,
            "receipts/future/ODYSSEY2_LAW_STORE.json": o2_origin,
            "receipts/future/ODYSSEY3_ADVERSARY.json": o3_origin,
            "receipts/future/PROPAGATION_STATE.json": prop_origin,
        },
        "odyssey2_n_laws": n_laws,
        "odyssey3_n_attacks": n_attacks,
        "propagate_applied_any": bool(
            isinstance(prop, dict)
            and any(
                (v or {}).get("applied")
                for v in (prop.get("consumers") or {}).values()
                if isinstance(v, dict)
            )
        ),
        "this_worktree_is_sparse": True,
        "unseen_is_not_absent": (
            "a path_taken of unseen_in_this_checkout means this sparse tree "
            "did not materialize the file; git ls-tree / the parent checkout may still hold it"
        ),
    }
    return FrontierBook(
        items=items,
        recovery=recovery,
        wake=wake,
        global_frontier=gf,
        scar_doc=scars if isinstance(scars, dict) else None,
        queue_identity=identity,
    )


# Public functional API (what HCLI / a future resident_api calls).

def next_work(available_lanes: Iterable[str] | str | None = None, *, book: FrontierBook | None = None) -> list[dict[str, Any]]:
    return (book or load_book()).next_work(available_lanes)


def is_idle(*, book: FrontierBook | None = None) -> bool:
    return (book or load_book()).is_idle()


def refill(
    available_lanes: Iterable[str] | str | None = None,
    *,
    exclude: Iterable[str] = (),
    book: FrontierBook | None = None,
) -> list[dict[str, Any]]:
    return (book or load_book()).refill(available_lanes, exclude=exclude)


def build(*, available_lanes: Iterable[str] | str | None = None, book: FrontierBook | None = None) -> Path:
    current = book or load_book()
    doc = current.to_doc(available_lanes=available_lanes)
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    book = load_book()
    blocked_lanes = (LANE_GPU_PROTECTED, LANE_ANE)
    available = [l for l in ALL_LANES if l not in blocked_lanes]
    units = book.next_work(available)
    idle = book.is_idle()
    if idle:
        raise SystemExit("selftest: is_idle() is True on the real frontier; the §140 failure condition fired")
    if not units:
        raise SystemExit("selftest: next_work() empty while GPU_PROTECTED and ANE are blocked")
    for unit in units:
        req = set(unit.get("required_lanes") or [])
        if req & {LANE_GPU_PROTECTED, LANE_ANE}:
            raise SystemExit(f"selftest: unsafe unit {unit.get('id')} requires {sorted(req)}")
        if str(unit.get("resource_class")) == "GPU_EXCLUSIVE":
            raise SystemExit(f"selftest: GPU_EXCLUSIVE unit leaked into CPU next_work: {unit.get('id')}")
    queued = book.items
    redundant = dict(next(i for i in book.items if i["kind"] == "NEXT_WORK" and not _item_sleeping(i, book.wake)))
    redundant["id"] = redundant["id"] + ".copy"
    red_decision = admit(redundant, queued=queued, book_items=book.items, scar_doc=book.scar_doc)
    if red_decision.get("admitted"):
        raise SystemExit("selftest: redundant unit was admitted")
    novel = {
        "id": "FT.CHILD_RESIDENT.novel-identity-dry-run-v2",
        "frontier": "CHILD_RESIDENT",
        "title": "Bind a child-resident identity receipt against the install contract without launching",
        "detail": "Distinct from the catalog dry-run: this targets identity-hash mismatch policy only.",
        "hypothesis_family": "child_resident_identity_hash_mismatch_policy",
        "expected_information_gain": INFO_HIGH,
        "description": "Novel identity-mismatch policy dry-run for a child resident, no process launch",
    }
    nov_decision = admit(novel, queued=queued, book_items=book.items, scar_doc=book.scar_doc)
    if not nov_decision.get("admitted"):
        raise SystemExit(f"selftest: novel unit refused: {nov_decision}")
    out = build(available_lanes=available, book=book)
    print(f"frontiers: {len(FRONTIER_NAMES)}")
    print(f"is_idle: {idle}")
    print(f"next_work (GPU_PROTECTED+ANE blocked): {len(units)}")
    print(f"sleeping: {len(book.sleeping_units())}")
    print(f"redundant admitted: {red_decision.get('admitted')} reason={red_decision.get('reason')}")
    print(f"novel admitted: {nov_decision.get('admitted')}")
    print(f"receipt: {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--next-work", action="store_true")
    ap.add_argument("--lanes", default=None, help="comma-separated available lanes")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    lanes = a.lanes
    if a.next_work:
        book = load_book()
        units = book.next_work(lanes)
        print(json.dumps({"n": len(units), "ids": [u["id"] for u in units], "is_idle": book.is_idle()}, indent=2))
        return 0
    print(build(available_lanes=lanes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
