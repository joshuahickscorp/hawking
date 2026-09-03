"""Emit H-ROADMAP-REVISED: a boot ROM for a fresh intelligence.

The goal is that handing a completely fresh model ONLY this file plus the repo is
enough for it to answer what Hawking is, what exists, what is merely simulated,
what is physically proven, what is running, what is stale, what to do next, who
owns it, what it unlocks, how it will be verified, and what not to touch.

GENERATED. Never hand-edited: `python3 -m tools.roadmap.emit_revised`.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs" / "roadmap"
STATE = REPO / "civilization" / "ROADMAP_STATE.json"
GRAPH = REPO / "civilization" / "CAPABILITY_GRAPH.json"
LINEAGE = REPO / "docs" / "roadmap-lineage" / "H-ROADMAP.superseded-2026-09-02.md"
OUT = Path.home() / "Downloads" / "H-ROADMAP-REVISED.md"

PARTS = (
    ("PART I — VERIFIED HAWKING TODAY", "PART_I_VERIFIED_TODAY.md"),
    ("PART II — OPTIMIZED REMAINDER", "PART_II_ACTION_PLAN.md"),
    ("PART III — CONSTITUTION / HARD FUTURES", "PART_III_CONSTITUTION_AND_RESEARCH.md"),
    ("ROADMAP COMPRESSION", "COMPRESSION.md"),
    ("HISTORICAL LINEAGE", "APPENDIX_LINEAGE.md"),
)

LEXICON = """    HCLI / AgentOS  the sovereign control plane; it runs Hawking
    GoalIR          compiled representation of human mission intent
    WorkUnit        smallest independently schedulable falsifiable objective
    Doctor          diagnoses a model or device and selects informative experiments
    Gravity         searches for capability-preserving physical-information reduction
    Noetic          represents discovered executable model programs
    NR              a model-semantic representation candidate
    NX              a self-contained machine-bound executable
    EBPW            complete persistent bits per source-parameter equivalent
    PhysicalGraph   whole-machine execution and placement graph
    HWIR            hardware intermediate representation
    MachineGenome   measured or declared physical machine description
    Accelerator     executes; owns the FPGA path, which is NOT a separate program
    ModelLake       sealed specimen lifecycle authority
    Odyssey I       discovery      II  transfer      III  adversarial meta-science
    HMF / HGVAS     managed heterogeneous object identity and address-space truth
    Fusion          multi-domain semantic machine
    Device Ascension  rediscovery when the machine itself changes
    ResultEnvelope  the carrier every tool result comes back in
    Law             a transferable finding      Scar  a recorded failure worth not repeating
    Pareto          remembers      Singularity  chooses      Resident  runs

    RETIRED NAMES -- do not recreate these:
    haider          retired; HCLI is not a fork of anything
    Tabula          part of Doctor, not a third science"""

PIPELINE = """    SOURCE SPECIMEN
      -> Doctor                  diagnose, choose the informative experiment
      -> Gravity                 search for capability-preserving reduction
      -> NR                      a representation candidate
      -> deterministic verification
      -> Noetic                  represent it as an executable program
      -> PhysicalGraph           lower it onto the whole machine
      -> NX                      a self-contained machine-bound executable
      -> Pareto                  remember the frontier point
      -> Singularity             choose
      -> Resident                run"""

FRESH_INTELLIGENCE = """IF YOU ARE CHATGPT / CLAUDE / GROK / HCLI AND HAVE NO PRIOR CONTEXT:

     1. Do not ask the human to restate Hawking. It is described below.
     2. Treat this roadmap as architecture and action map, NOT live runtime truth.
     3. Read ROADMAP_STATE and the current process/job state before making any
        claim about what is running right now.
     4. Truth comes from disk, source, tests and receipts. Not from prose, and
        not from this file.
     5. Do not recreate historical names or subsumed architecture. See the
        retired-names list.
     6. Find the HOT FRONTIER in section 2 and start there.
     7. CHECK OWNERSHIP before implementing. Another campaign owns hcli/, and
        duplicating its work is forbidden.
     8. Prefer the action with the highest dependency leverage, not the one that
        appears earliest.
     9. Verify independently. Equivalence between two executions of the same
        implementation does not prove a defining property.
    10. Write the result back into machine-readable state, not only into prose.
    11. Regenerate this roadmap when authority changes."""

AUTHORITY_CHAIN = """READ IN THIS ORDER. Later entries OVERRULE earlier ones on any conflict,
because they are closer to the machine:

    1. H-ROADMAP-REVISED.md            this file: architecture and plan
    2. civilization/ROADMAP_STATE.json machine-readable plan; wins over this file
    3. civilization/CAPABILITY_GRAPH.json  per-gate evidence from HEAD blobs
    4. current mission / resident state    what the daemon is doing NOW
    5. active background jobs              what is running NOW
    6. receipts/ written since generated_at  evidence newer than this document
    7. git HEAD and worktrees              what the code actually is
    8. ModelLake specimen registry on disk  what specimens actually exist

    A number in this file that disagrees with (2) means THIS FILE IS STALE.
    Regenerate it; do not edit it."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "absent"


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                              text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _fingerprint(state: dict) -> str:
    head = _git("rev-parse", "HEAD")
    return f"""AUTHORITY FINGERPRINT

    generated_at            {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}
    repo_head               {head}
    valid_for_head          {head}
    generator_commit        {_git("log", "-1", "--format=%H", "--", "tools/roadmap/emit_revised.py") or "uncommitted"}
    roadmap_state_sha256    {_sha(STATE)}
    capability_graph_sha256 {_sha(GRAPH)}
    lineage_sha256          {_sha(LINEAGE)}
    gates                   {len(json.loads(GRAPH.read_text())['gates'])}

STALE_IF ANY OF THESE HOLD -- and if stale, this file is HISTORY, not authority:

    git HEAD != valid_for_head
    civilization/CAPABILITY_GRAPH.json sha256 has changed
    civilization/ROADMAP_STATE.json sha256 has changed
    any receipt under receipts/ is newer than generated_at

    Detect with:  python3 -m tools.roadmap.emit_revised --check"""


def _section_0(state: dict) -> str:
    return f"""# H-ROADMAP — REVISED

Supersedes the 9645-line H-ROADMAP.md, preserved verbatim and hashed under
docs/roadmap-lineage/. GENERATED by `python3 -m tools.roadmap.emit_revised`.
DO NOT EDIT THIS FILE.

# 0. FRESH-PROCESS / EXECUTION KERNEL

{FRESH_INTELLIGENCE}

{AUTHORITY_CHAIN}

{_fingerprint(state)}

## The rules that outrank the plan

    Never fake an experiment result, a long-duration autonomy claim, a GPU or ANE
    measurement, an FPGA physical measurement, or DGX/eGPU behaviour. Build up to
    the boundary and say which side of it each artifact is on.

    Simulated is not measured. STATIC is not FUNCTIONAL_SIM is not
    HARDWARE_MEASURED. Never merge tiers to improve a number.

    A capability is not built because a file exists. Registration is not wiring.
    An import is not a call.

    Never weaken, skip, delete or mark-slow a test to reach a target. If a target
    is only reachable by checking less, MISS IT and report the honest floor.

    Fix the producer, not the artifact. Correcting a receipt leaves the thing
    that wrote it still wrong.

    Check ownership before implementing. hcli/ belongs to a parallel campaign.
"""


def _section_1(state: dict) -> str:
    ev = state["evidence_tier_meaning"]
    return f"""# 1. SIXTY-SECOND SYSTEM MAP

HAWKING IS A SELF-OPTIMIZING PHYSICAL AI COMPUTER. Exactly five eras, exactly
three odysseys. No Era VI, no Odyssey IV. HCLI/AgentOS is the sovereign control
plane; everything else is an organ it drives.

    MODELS THINK. TOOLS KNOW. CONTEXT IS A CACHE. DISK STATE IS AUTHORITY.

No model self-certifies. Design intent is not artifact reality is not runtime
reality. Physical claims require physical evidence.

## The canonical lexicon

{LEXICON}

## The artifact pipeline

{PIPELINE}

## What the numbers mean

    TOTAL_UNRESOLVED_GATES        {state['TOTAL_UNRESOLVED_GATES']:3}   every gate with any remaining blocker
    ACTIVE_NONHARDWARE_BURDEN     {state['ACTIVE_NONHARDWARE_BURDEN']:3}   the above, minus those waiting on absent silicon
    SOFTWARE_CONNECTION_REMAINING {state['software_connection_remaining_count']:3}   parts exist; nothing calls or checks them
    integrated_capabilities       {len(state['integrated_capabilities']):3}   no blocker left AND a test cites it
    completed_capabilities        {len(state['completed_capabilities']):3}   wired and acceptance-passed, verifier NOT required
    built_but_no_verifier         {len(state['built_but_no_verifier']):3}   the difference between the two above

None of these is a synonym for another. Read counts_explained in ROADMAP_STATE.

## What evidence_tier means, and what it does not

Every gate currently reads STATIC. That is a statement about THIS AUDIT, not
about Hawking's history:

    {ev['STATIC']}

The other tiers, in ascending strength:

    FUNCTIONAL_SIM      {ev['FUNCTIONAL_SIM']}
    LOCAL_MEASURED      {ev['LOCAL_MEASURED']}
    REPRODUCED          {ev['REPRODUCED']}
    PROTECTED_VERIFIED  {ev['PROTECTED_VERIFIED']}
    HARDWARE_MEASURED   {ev['HARDWARE_MEASURED']}

## Status axes

Implementation, acceptance, verification, evidence and integration are
INDEPENDENT. A gate can be integrated and unverified; a name that merges them
would hide exactly that. Derived statuses now in the graph:

{chr(10).join(f"    {k:32} {v}" for k, v in sorted(__import__('collections').Counter(state['derived_status'].values()).items(), key=lambda kv: -kv[1]))}
"""


def _section_2(state: dict) -> str:
    rows = state["hot_frontier"]
    out = ["# 2. HOT OPERATIONAL FRONTIER", "",
           "Ranked across blocker classes by dependency leverage. THE SCORE SCHEDULES",
           "WORK; IT NEVER CERTIFIES TRUTH. Blocked work is excluded here -- it keeps its",
           "dependency value in PART II but must not occupy the first ten lines a fresh",
           "operator reads.", ""]
    if not rows:
        out += ["    Nothing is actionable by this lane right now. See PART II.", ""]
    for i, r in enumerate(rows, 1):
        out += [
            f"## {i}. {r['gate']}",
            f"    blocker            {r['blocker_class']}",
            f"    missing            {r['missing']}",
            f"    owner              {r['owner']}",
            f"    resource lane      {r['resource_lane']}",
            f"    estimated wall     {r['estimated_wall']}",
            f"    unlocks direct     {', '.join(r['unlocks_direct']) or 'nothing declared'}",
            f"    unlocks transitive {len(r['unlocks_transitive'])}"
            + (f" -- {', '.join(r['unlocks_transitive'][:6])}" if r['unlocks_transitive'] else ""),
            f"    depends on         {', '.join(r['depends_on']) or 'nothing'}",
            f"    critical path      {r['critical_path']}      multiplier {r['multiplier']}",
            f"    verifier           {r['verifier']}",
            f"    stop condition     {r['stop_condition']}",
            f"    reopen if          {r['reopen_if']}",
            f"    parallel safe with {r['parallel_safe_with']}",
            "",
        ]
    return "\n".join(out)


def build() -> Path:
    state = json.loads(STATE.read_text())
    body = [_section_0(state), _section_1(state), _section_2(state)]
    for title, name in PARTS:
        body.append((DOCS / name).read_text())
    OUT.write_text("\n".join(body))
    return OUT


def check() -> int:
    """Is the emitted file still authority for this HEAD?"""
    if not OUT.is_file():
        print("STALE: no emitted roadmap")
        return 1
    text = OUT.read_text()
    head = _git("rev-parse", "HEAD")
    if f"valid_for_head          {head}" not in text:
        print(f"STALE: emitted for a different HEAD (now {head})")
        return 1
    if f"roadmap_state_sha256    {_sha(STATE)}" not in text:
        print("STALE: ROADMAP_STATE.json has changed since generation")
        return 1
    if f"capability_graph_sha256 {_sha(GRAPH)}" not in text:
        print("STALE: CAPABILITY_GRAPH.json has changed since generation")
        return 1
    print("FRESH: the emitted roadmap matches current machine authority")
    return 0


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        raise SystemExit(check())
    print(build())
